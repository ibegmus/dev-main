"""Durable daily quotas and private, one-time rewarded feedback."""
from datetime import datetime, timedelta, timezone
import secrets
import db

ZONE = timezone(timedelta(hours=5))

def day():
    return datetime.now(ZONE).date().isoformat()

def init(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS request_usage (
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, day TEXT NOT NULL,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        completed INTEGER NOT NULL DEFAULT 0)''')
    conn.execute('CREATE INDEX IF NOT EXISTS usage_owner_day ON request_usage(owner, day)')
    conn.execute('''CREATE TABLE IF NOT EXISTS feedback (
        user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        conversation_id TEXT NOT NULL, stars INTEGER NOT NULL CHECK(stars BETWEEN 1 AND 5),
        comment TEXT NOT NULL, bonus_day TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)''')

def recover():
    with db.get_conn() as conn:
        conn.execute('DELETE FROM request_usage WHERE completed = 0 OR day < ?', (day(),))

def _status(conn, owner, user_id):
    admin = bool(user_id and conn.execute("SELECT 1 FROM users WHERE id=? AND role='admin'", (user_id,)).fetchone())
    feedback = conn.execute('SELECT bonus_day FROM feedback WHERE user_id=?', (user_id,)).fetchone() if user_id else None
    limit = 5 + (5 if feedback and feedback['bonus_day'] == day() else 0)
    used = conn.execute('SELECT count(*) FROM request_usage WHERE owner=? AND day=?', (owner, day())).fetchone()[0]
    return dict(admin=admin, limit=None if admin else limit, remaining=None if admin else max(0,limit-used),
                feedback_submitted=bool(feedback), bonus_available=bool(user_id and not feedback), day=day(), timezone='Asia/Yekaterinburg')

def status(owner, user_id):
    with db.get_conn() as conn:
        return _status(conn, owner, user_id)

def reserve(owner, user_id):
    with db.get_conn() as conn:
        conn.execute('BEGIN IMMEDIATE')
        state = _status(conn, owner, user_id)
        if not state['admin'] and state['remaining'] <= 0:
            message = 'Дневной лимит исчерпан. Новые 5 запросов появятся в полночь по Екатеринбургу.'
            if state['bonus_available']:
                message += ' За первый отзыв с комментарием можно получить ещё 5 запросов.'
            elif not user_id:
                message += ' Войдите в аккаунт для сохранения бесед и возможности оставить отзыв.'
            raise ValueError(message)
        if state['admin']:
            return None
        token = secrets.token_hex(24)
        conn.execute('INSERT INTO request_usage(id,owner,day,user_id) VALUES (?,?,?,?)', (token,owner,day(),user_id))
        return token

def finish(token, success):
    if not token:
        return
    with db.get_conn() as conn:
        if success:
            conn.execute('UPDATE request_usage SET completed=1 WHERE id=?', (token,))
        else:
            conn.execute('DELETE FROM request_usage WHERE id=?', (token,))

def submit_feedback(user_id, conversation_id, stars, comment):
    comment = comment.strip()
    if not 10 <= len(comment) <= 2000:
        raise ValueError('Напишите комментарий от 10 до 2000 символов.')
    with db.get_conn() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT 1 FROM feedback WHERE user_id=?', (user_id,)).fetchone():
            raise ValueError('Вы уже отправили отзыв. Бонус выдаётся один раз на аккаунт.')
        if not conn.execute('''SELECT 1 FROM conversations c JOIN messages m ON m.conversation_id=c.id
            WHERE c.id=? AND c.user_id=? AND m.user_id=? LIMIT 1''', (conversation_id,user_id,user_id)).fetchone():
            raise ValueError('Оценить можно свою беседу, в которой уже есть ответ ИИ.')
        conn.execute('INSERT INTO feedback(user_id,conversation_id,stars,comment,bonus_day) VALUES (?,?,?,?,?)',
                     (user_id,conversation_id,stars,comment,day()))

def feedback_list(before=2147483647):
    with db.get_conn() as conn:
        return [dict(r) for r in conn.execute('SELECT user_id,stars,comment,created_at FROM feedback WHERE user_id<? ORDER BY created_at DESC, user_id DESC LIMIT 50', (before,))]
