"""Local password policy and versioned hashing; no passwords leave the server."""
import hashlib
import hmac
import secrets
import unicodedata

ITERATIONS = 600_000
MIN_LENGTH = 12
MAX_LENGTH = 128
# A small local denylist, not a complete database of leaked passwords.
COMMON_PASSWORDS = frozenset({
    'password', 'password123', 'password1234', 'password12345', 'password123456',
    'password123!', 'password123!@#', 'qwerty', 'qwerty123', 'qwerty123456',
    'qwertyuiop', 'qwertyuiop123', '123456', '12345678', '123456789',
    '1234567890', '123456789012', '1234567890123456', '123456654321',
    '1q2w3e4r5t6y', '1qaz2wsx3edc', 'qazwsxedcrfv', 'admin12345678',
    'administrator', 'letmein', 'iloveyou', 'welcome12345', 'changeme12345',
    'пароль', 'пароль123456', 'пароль123456789', 'йцукенгшщзхъ',
    'terramind123', 'terramind1234', 'terramind12345', 'pochva-ai.tech',
})


def password_problem(password, email=''):
    if len(password) < MIN_LENGTH:
        return 'Используйте не менее 12 символов — например, длинную фразу из нескольких слов.'
    if len(password) > MAX_LENGTH:
        return 'Пароль должен содержать не более 128 символов.'
    folded = unicodedata.normalize('NFKC', password).casefold()
    compact = ''.join(c for c in folded if c.isalnum())
    common = {''.join(c for c in item if c.isalnum()) for item in COMMON_PASSWORDS}
    if compact in common or len(set(password)) < 4:
        return 'Этот пароль слишком распространённый или повторяющийся. Выберите уникальную длинную фразу.'
    if email and folded in {email.casefold(), email.partition('@')[0].casefold()}:
        return 'Пароль не должен совпадать с вашим email или его именем.'
    return None


def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt), ITERATIONS).hex()
    return f'pbkdf2_sha256${ITERATIONS}${salt}${digest}'


def password_matches(stored, password):
    try:
        parts = stored.split('$')
        if len(parts) == 2:
            salt, expected = parts
            iterations = 200_000  # Original database format.
        elif len(parts) == 4 and parts[0] == 'pbkdf2_sha256':
            _, count, salt, expected = parts
            iterations = int(count)
            if not 200_000 <= iterations <= 1_000_000:
                return False
        else:
            return False
        if len(salt) != 32 or len(expected) != 64:
            return False
        digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt), iterations).hex()
        return hmac.compare_digest(digest, expected)
    except (ValueError, TypeError, AttributeError):
        return False


def needs_upgrade(stored):
    return not stored.startswith(f'pbkdf2_sha256${ITERATIONS}$')
