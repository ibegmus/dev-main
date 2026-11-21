"""Defence-in-depth HTTP headers for the same-origin web application."""
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse

SENSITIVE = ('/register', '/login', '/logout', '/me', '/account', '/history', '/ask', '/conversations', '/feedback', '/admin')


class SecurityHeaders:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        async def secured(message):
            if message['type'] == 'http.response.start':
                headers = MutableHeaders(scope=message)
                headers['X-Content-Type-Options'] = 'nosniff'
                headers['X-Frame-Options'] = 'DENY'
                headers['Referrer-Policy'] = 'no-referrer'
                headers['Permissions-Policy'] = 'geolocation=(), microphone=(), payment=()'
                headers['Content-Security-Policy'] = (
                    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
                    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
                )
                path = scope['path']
                if any(path == p or path.startswith(p + '/') for p in SENSITIVE):
                    headers['Cache-Control'] = 'no-store'
            await send(message)
        if scope['method'] in {'POST', 'PUT', 'PATCH', 'DELETE'}:
            path = scope['path'].rstrip('/') or '/'
            limit = (8 * 1024 * 1024 if path in {'/ask', '/ask/jobs'} else
                     11 * 1024 * 1024 if path == '/upload' else
                     2 * 1024 * 1024 if path == '/knowledge' else 16 * 1024)
            too_large = JSONResponse({'detail': 'Запрос слишком большой. Уменьшите размер файла или текста.'}, status_code=413)
            headers = dict(scope.get('headers', []))
            try:
                declared = int(headers.get(b'content-length', b'0'))
            except ValueError:
                declared = 0
            if declared > limit:
                return await too_large(scope, receive, secured)
            body = bytearray()
            total = 0
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                total += len(message.get('body', b''))
                if total > limit:
                    return await too_large(scope, receive, secured)
                body.extend(message.get('body', b''))
                if not message.get('more_body', False):
                    break

            replayed = False
            async def replay():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    payload = bytes(body)
                    body.clear()
                    return {'type': 'http.request', 'body': payload, 'more_body': False}
                return await receive()
            return await self.app(scope, replay, secured)
        await self.app(scope, receive, secured)
