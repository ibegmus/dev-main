import sqlite3
import json
import hashlib
import secrets
import os
from contextlib import contextmanager

from storage import default_database_path, prepare_database_path
from password_security import hash_password, password_matches, needs_upgrade

DB_FILE = default_database_path()
SESSION_DAYS = 30
MAX_SAVED_CONVERSATIONS = 50


@contextmanager
def get_conn():
    """Commit/roll back the transaction and always close the SQLite handle."""
    conn = sqlite3.connect(DB_FILE, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            yield conn
    finally:
        conn.close()


def init_db():
    """Создаёт таблицы, если их ещё нет. Безопасно вызывать при каждом старте."""
    prepare_database_path(DB_FILE)
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            email         TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS monoliths (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            region      TEXT,
            year        INTEGER,
            soil_type   TEXT,
            humus       REAL,
            ph          REAL,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token      TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        -- История диалогов (вместо chat_history.txt)
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER REFERENCES users(id) ON DELETE SET NULL,
            question   TEXT NOT NULL,
            answer     TEXT NOT NULL,
            intent     TEXT,
            seconds    REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        -- Знания для нейросети (первоисточник; в Chroma лежит только индекс)
        CREATE TABLE IF NOT EXISTS knowledge (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.execute("BEGIN IMMEDIATE")
        additions = {
            "users": {"terms_version": "TEXT", "terms_accepted_at": "TEXT", "role": "TEXT NOT NULL DEFAULT 'user'"},
            "sessions": {"expires_at": "TEXT"},
            "messages": {"conversation_id": "TEXT", "has_image": "INTEGER NOT NULL DEFAULT 0"},
        }
        for table, columns in additions.items():
            present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for column, declaration in columns.items():
                if column not in present:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            image_description TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        conversation_columns = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
        if 'soil_case' not in conversation_columns:
            conn.execute("ALTER TABLE conversations ADD COLUMN soil_case TEXT NOT NULL DEFAULT '{}'")
        import usage
        usage.init(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS conversations_owner ON conversations(user_id, updated_at)")
        conn.execute("UPDATE sessions SET expires_at = datetime(created_at, '+30 days') WHERE expires_at IS NULL")
        # Old versions had no conversation grouping. Preserve each exchange in
        # its own saved conversation rather than inventing relationships.
        old = conn.execute("""SELECT id, user_id, question, created_at FROM messages
            WHERE conversation_id IS NULL AND user_id IS NOT NULL
            AND EXISTS (SELECT 1 FROM users WHERE users.id = messages.user_id)""").fetchall()
        for row in old:
            cid = secrets.token_urlsafe(32)
            title = ' '.join(row['question'].split())[:100] or 'Сохранённый вопрос'
            conn.execute("""INSERT INTO conversations(id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)""", (cid, row['user_id'], title, row['created_at'], row['created_at']))
            conn.execute("UPDATE messages SET conversation_id = ?, has_image = ? WHERE id = ?",
                         (cid, int('[с фото]' in row['question']), row['id']))


# ---------- Пользователи ----------

def _hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 200_000
    ).hex()


def create_user(name: str, email: str, password: str, terms_version=None):
    """Return a new user ID, or None for a duplicate address."""
    stored = hash_password(password)
    try:
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO users (name, email, password_hash, terms_version, terms_accepted_at)
                VALUES (?, ?, ?, ?, CASE WHEN ? IS NULL THEN NULL ELSE CURRENT_TIMESTAMP END)""",
                (name.strip(), email.lower().strip(), stored, terms_version, terms_version),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def _password_matches(stored, password):
    return password_matches(stored, password)


def verify_password(user_id, password):
    with get_conn() as conn:
        user = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
        return bool(user and _password_matches(user['password_hash'], password))


def get_user(user_id):
    with get_conn() as conn:
        row = conn.execute("SELECT id, name, email, role FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def login(email: str, password: str):
    """Return a session token with a fixed thirty-day lifetime."""
    with get_conn() as conn:
        user = conn.execute("SELECT id, password_hash FROM users WHERE email = ?",
                            (email.lower().strip(),)).fetchone()
        if user is None:
            hash_password(password)
            return None
        if not _password_matches(user['password_hash'], password):
            return None
        token = secrets.token_hex(32)
        # An account deleted during password verification must not reappear.
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user['id'],)).fetchone()
        if current is None or current['password_hash'] != user['password_hash']:
            return None
        if needs_upgrade(user['password_hash']):
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(password), user['id']))
        conn.execute("DELETE FROM sessions WHERE expires_at <= CURRENT_TIMESTAMP")
        conn.execute("""INSERT INTO sessions (token, user_id, expires_at)
                        VALUES (?, ?, datetime('now', '+30 days'))""", (token, user['id']))
        conn.execute("""DELETE FROM sessions WHERE user_id = ? AND token NOT IN
                        (SELECT token FROM sessions WHERE user_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 20)""",
                     (user['id'], user['id']))
        return token


def change_password(user_id, current_password, new_password):
    """Atomically change credentials and revoke all sessions, returning a new one."""
    with get_conn() as conn:
        user = conn.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,)).fetchone()
        if user is None or not password_matches(user['password_hash'], current_password):
            return None
        replacement = hash_password(new_password)
        conn.execute('BEGIN IMMEDIATE')
        changed = conn.execute('UPDATE users SET password_hash = ? WHERE id = ? AND password_hash = ?',
                               (replacement, user_id, user['password_hash'])).rowcount
        if not changed:
            return None
        conn.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
        token = secrets.token_hex(32)
        conn.execute("INSERT INTO sessions(token, user_id, expires_at) VALUES (?, ?, datetime('now', '+30 days'))", (token, user_id))
        return token


def get_user_by_token(token: str):
    if not token:
        return None
    with get_conn() as conn:
        return conn.execute(
            """SELECT users.id, users.name, users.email, users.role
               FROM sessions JOIN users ON users.id = sessions.user_id
               WHERE sessions.token = ? AND sessions.expires_at > CURRENT_TIMESTAMP""",
            (token,),
        ).fetchone()


def revoke_session(token):
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def delete_user(user_id):
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return conn.execute("DELETE FROM users WHERE id = ?", (user_id,)).rowcount > 0


# ---------- Сообщения ----------

def save_message(user_id, question, answer, intent, seconds):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, question, answer, intent, seconds) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, question, answer, intent, seconds),
        )


def get_history(user_id: int, limit: int = 50):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT question, answer, created_at, has_image FROM messages "
            "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            if value.pop('has_image') and '[с фото]' not in value['question']:
                value['question'] += ' [с фото]'
            result.append(value)
        return result


# ---------- Знания ----------

def add_knowledge(title: str, content: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO knowledge (title, content) VALUES (?, ?)", (title, content)
        )
        return cur.lastrowid


def list_knowledge():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM knowledge ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

# ---------- Монолиты ----------

def add_monolith(name, region=None, year=None, soil_type=None,
                 humus=None, ph=None, description=None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO monoliths (name, region, year, soil_type, humus, ph, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, region, year, soil_type, humus, ph, description),
        )
        return cur.lastrowid


def list_monoliths():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM monoliths ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def find_monoliths(query: str, limit: int = 5):
    """Ищем в Python, потому что LIKE в SQLite не понимает регистр кириллицы."""
    q = query.lower().strip()
    found = []
    for m in list_monoliths():
        text = f"{m['name']} {m['region'] or ''} {m['soil_type'] or ''}".lower()
        if q in text:
            found.append(m)
    return found[:limit]


# ---------- Сохранённые беседы ----------

def create_conversation(conversation_id, user_id, title):
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not conn.execute('SELECT 1 FROM users WHERE id = ?', (user_id,)).fetchone():
            return False
        if conn.execute('SELECT COUNT(*) FROM conversations WHERE user_id = ?', (user_id,)).fetchone()[0] >= MAX_SAVED_CONVERSATIONS:
            return False
        try:
            conn.execute('INSERT INTO conversations(id, user_id, title) VALUES (?, ?, ?)',
                         (conversation_id, user_id, ' '.join(title.split())[:100] or 'Новая беседа'))
        except sqlite3.IntegrityError:
            return False
        return True


def get_conversation(conversation_id, user_id):
    with get_conn() as conn:
        row = conn.execute('SELECT *, id AS conversation_id FROM conversations WHERE id = ? AND user_id = ?',
                           (conversation_id, user_id)).fetchone()
        return dict(row) if row else None


def list_conversations(user_id):
    with get_conn() as conn:
        return [dict(row) for row in conn.execute(
            'SELECT id AS conversation_id, title, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC, rowid DESC',
            (user_id,)).fetchall()]


def get_conversation_turns(conversation_id, user_id, before=None, limit=100):
    limit = min(max(int(limit), 1), 100)
    with get_conn() as conn:
        if not conn.execute('SELECT 1 FROM conversations WHERE id = ? AND user_id = ?',
                            (conversation_id, user_id)).fetchone():
            return None
        params = [conversation_id, user_id]
        where = 'conversation_id = ? AND user_id = ?'
        if before is not None:
            where += ' AND id < ?'
            params.append(int(before))
        rows = conn.execute(f'''SELECT id, question, answer, has_image FROM messages
            WHERE {where} ORDER BY id DESC LIMIT ?''', [*params, limit + 1]).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        turns = [dict(row) for row in reversed(rows)]
        for turn in turns:
            turn['has_image'] = bool(turn['has_image'])
        return {'turns': turns, 'has_more': has_more,
                'next_before': turns[0]['id'] if has_more and turns else None}


def get_conversation_memory(conversation_id, user_id):
    # All reads share one transaction snapshot, including a concurrent deletion.
    with get_conn() as conn:
        conn.execute('BEGIN')
        conversation = conn.execute('SELECT image_description, soil_case FROM conversations WHERE id = ? AND user_id = ?',
                                    (conversation_id, user_id)).fetchone()
        if conversation is None:
            return None
        rows = conn.execute('''SELECT question, answer, has_image FROM messages
            WHERE conversation_id = ? AND user_id = ? ORDER BY id DESC LIMIT 8''',
                            (conversation_id, user_id)).fetchall()
        turns = []
        remaining = 24000
        for row in rows:
            turn = {'question': row['question'][:4000], 'answer': row['answer'][:6000],
                    'has_image': bool(row['has_image'])}
            length = len(turn['question']) + len(turn['answer'])
            if length > remaining:
                break
            remaining -= length
            turns.append(turn)
        memory = {'history': list(reversed(turns)), 'image_description': conversation['image_description'][:4000]}
        try:
            case = json.loads(conversation['soil_case'])
            if isinstance(case, dict) and case:
                memory['soil_case'] = case
        except (TypeError, ValueError):
            pass
        return memory


def save_conversation_turn(conversation_id, user_id, question, answer, intent=None, seconds=None,
                           has_image=False, image_description=None, soil_case=None, clear_photo=False):
    with get_conn() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('''SELECT 1 FROM conversations JOIN users ON conversations.user_id = users.id
            WHERE conversations.id = ? AND conversations.user_id = ?''', (conversation_id, user_id)).fetchone():
            return False
        conn.execute('''INSERT INTO messages(user_id, question, answer, intent, seconds, conversation_id, has_image)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (user_id, question[:4000], answer[:16000], intent, seconds, conversation_id, int(has_image)))
        if has_image or clear_photo:
            conn.execute('UPDATE conversations SET image_description = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                         ((image_description or '')[:4000], conversation_id))
        else:
            conn.execute('UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?', (conversation_id,))
        if isinstance(soil_case, dict):
            conn.execute('UPDATE conversations SET soil_case = ? WHERE id = ?',
                         (json.dumps(soil_case, ensure_ascii=False), conversation_id))
        return True


def delete_conversation(conversation_id, user_id):
    with get_conn() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM conversations WHERE id = ? AND user_id = ?',
                            (conversation_id, user_id)).fetchone():
            return False
        conn.execute('DELETE FROM messages WHERE conversation_id = ? AND user_id = ?', (conversation_id, user_id))
        conn.execute('DELETE FROM conversations WHERE id = ? AND user_id = ?', (conversation_id, user_id))
        return True
