"""Session request checks and bounded, process-local abuse limits."""
import hashlib
import hmac
import secrets
import threading
import time
from collections import OrderedDict, deque
from urllib.parse import urlsplit

from fastapi import HTTPException

SESSION_COOKIE = 'terramind_session'
GUEST_COOKIE = 'terramind_guest'
SESSION_AGE = 30 * 24 * 60 * 60
TERMS_VERSION = '2026-11-06.feedback'
_GUEST_SECRET = secrets.token_bytes(32)


def secure_cookie(request):
    return request.url.hostname not in {'localhost', '127.0.0.1', '::1', 'testserver'}


def check_browser_mutation(request):
    origin = request.headers.get('origin')
    if origin:
        try:
            parsed = urlsplit(origin)
        except ValueError as exc:
            raise HTTPException(403, 'Некорректный источник запроса.') from exc
        if parsed.netloc != request.headers.get('host') or parsed.scheme not in {'http', 'https'}:
            raise HTTPException(403, 'Запрос с другого сайта запрещён.')
        if secure_cookie(request) and parsed.scheme != 'https':
            raise HTTPException(403, 'Требуется защищённое соединение.')
    if request.cookies.get(SESSION_COOKIE) and request.headers.get('x-terramind-request') != '1':
        raise HTTPException(403, 'Обновите страницу и повторите действие.')


def guest_identity(request, response):
    cookie = request.cookies.get(GUEST_COOKIE, '')
    try:
        value, signature = cookie.split('.')
        expected = hmac.new(_GUEST_SECRET, value.encode(), hashlib.sha256).hexdigest()
        if len(value) == 43 and hmac.compare_digest(expected, signature):
            return value
    except ValueError:
        pass
    value = secrets.token_urlsafe(32)
    signature = hmac.new(_GUEST_SECRET, value.encode(), hashlib.sha256).hexdigest()
    response.set_cookie(GUEST_COOKIE, value + '.' + signature, httponly=True,
                        secure=secure_cookie(request), samesite='lax', max_age=SESSION_AGE)
    return value


class WindowLimiter:
    def __init__(self, max_keys=2048, clock=time.monotonic):
        self.max_keys = max_keys
        self.clock = clock
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key, *, limit, seconds, message):
        # Hash keys so even the temporary limiter does not retain raw emails/IPs.
        key = hashlib.sha256(str(key).encode()).digest()
        with self._lock:
            now = self.clock()
            entries = self._entries.pop(key, deque())
            while entries and entries[0] <= now - seconds:
                entries.popleft()
            self._entries[key] = entries
            if len(entries) >= limit:
                retry = max(1, int(seconds - (now - entries[0])) + 1)
                raise HTTPException(429, message, headers={'Retry-After': str(retry)})
            entries.append(now)
            while len(self._entries) > self.max_keys:
                self._entries.popitem(last=False)
