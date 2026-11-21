"""Account isolation, durable conversations, and deletion during inference."""
import base64
import copy
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import api
import db

MUTATION = {'X-TerraMind-Request': '1'}
PASSWORD = 'Example-password-123'


class AccountAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        self.calls = []
        self.inflight = 0
        self.max_inflight = 0

        def answer(text, image, memory=None):
            self.calls.append((text, image, copy.deepcopy(memory)))
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            self.started.set()
            try:
                if not self.release.wait(8):
                    raise RuntimeError('test did not release model')
                return {'answer': 'Ответ: ' + text, 'intent': 'science',
                        'image_description': 'Тёмный горизонт над белым слоем' if image else ''}
            finally:
                self.inflight -= 1

        self.patches = [patch.object(db, 'DB_FILE', str(Path(self.temp.name) / 'accounts.db')),
                        patch.object(api, 'preload_reranker'),
                        patch.object(api, 'run_agent', side_effect=answer)]
        for item in self.patches:
            item.start()
        self.client = TestClient(api.app, base_url='http://localhost')
        self.client.__enter__()
        self.others = []

    def tearDown(self):
        self.release.set()
        for client in self.others:
            client.close()
        self.client.__exit__(None, None, None)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def other_client(self, secure=False):
        client = TestClient(api.app, base_url='https://api.pochva-ai.tech' if secure else 'http://localhost')
        client.portal = self.client.portal
        self.others.append(client)
        return client

    def register(self, client=None, email='first@example.invalid'):
        client = client or self.client
        response = client.post('/register', json={
            'name': 'Пользователь', 'email': email, 'password': PASSWORD,
            'terms_accepted': True}, headers=MUTATION)
        self.assertIn(response.status_code, (200, 201), response.text)
        self.assertEqual(response.json()['user']['email'], email)
        return response

    def submit(self, text='Что такое чернозём?', cid=None, client=None, image=None):
        body = {'text': text}
        if cid:
            body['conversation_id'] = cid
        if image:
            body['image'] = image
        return (client or self.client).post('/ask/jobs', json=body, headers=MUTATION)

    def done(self, job, client=None):
        client = client or self.client
        for _ in range(500):
            response = client.get('/ask/jobs/' + job)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            if result['status'] in ('done', 'failed'):
                self.assertEqual(result['status'], 'done', result)
                return result
            time.sleep(.01)
        self.fail('inference did not finish')

    def new_chat(self, text='Что такое чернозём?', client=None, image=None):
        response = self.submit(text, client=client, image=image)
        self.assertEqual(response.status_code, 202, response.text)
        result = response.json()
        self.done(result['job_id'], client)
        return result

    def test_register_cookie_login_logout_and_terms_are_recorded(self):
        response = self.register()
        cookie = response.headers.get('set-cookie', '').lower()
        self.assertIn('httponly', cookie)
        self.assertIn('samesite=lax', cookie)
        self.assertIn('max-age=', cookie)
        self.assertEqual(self.client.get('/me').json()['user']['email'], 'first@example.invalid')
        uid = response.json()['user']['id']
        with db.get_conn() as conn:
            user = conn.execute('SELECT * FROM users WHERE id=?', (uid,)).fetchone()
            self.assertTrue(user['terms_version'])
            self.assertTrue(user['terms_accepted_at'])
            self.assertNotEqual(user['password_hash'], PASSWORD)
        self.assertEqual(self.client.post('/logout', headers=MUTATION).status_code, 204)
        self.assertEqual(self.client.get('/me').status_code, 401)
        response = self.client.post('/login', json={'email': 'FIRST@example.invalid', 'password': PASSWORD}, headers=MUTATION)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get('/me').status_code, 200)

    def test_cookie_is_secure_on_public_https_host(self):
        client = self.other_client(secure=True)
        response = self.register(client, 'secure@example.invalid')
        self.assertIn('secure', response.headers.get('set-cookie', '').lower())
        self.assertEqual(client.get('/me').status_code, 200)

    def test_registration_rejects_missing_terms_and_invalid_input(self):
        valid = {'name': 'Test', 'email': 'a@example.invalid', 'password': PASSWORD, 'terms_accepted': True}
        for overrides in ({'terms_accepted': False}, {'password': 'short'}, {'email': 'broken'}, {'name': '   '}):
            body = dict(valid, **overrides)
            response = self.client.post('/register', json=body, headers=MUTATION)
            self.assertIn(response.status_code, (400, 422), response.text)
        with db.get_conn() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM users').fetchone()[0], 0)

    def test_cookie_mutations_reject_missing_header_and_foreign_origin(self):
        self.register()
        no_header = self.client.post('/ask/jobs', json={'text': 'q'})
        self.assertEqual(no_header.status_code, 403, no_header.text)
        foreign = self.client.post('/ask/jobs', json={'text': 'q'}, headers={**MUTATION, 'Origin': 'https://foreign.invalid'})
        self.assertEqual(foreign.status_code, 403, foreign.text)
        self.assertEqual(self.calls, [])

    def test_invalid_token_does_not_silently_become_guest(self):
        response = self.client.post('/ask/jobs', json={'text': 'q'}, headers={**MUTATION, 'x-token': 'not-a-session'})
        self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(self.calls, [])

    def test_account_chat_and_job_are_inaccessible_to_other_account_or_guest(self):
        self.register()
        first = self.new_chat()
        other = self.other_client()
        self.register(other, 'other@example.invalid')
        guest = self.other_client()
        cid = first['conversation_id']
        for client in (other, guest):
            denied = (401, 404, 410) if client is guest else (401, 404)
            self.assertIn(client.get('/conversations/' + cid).status_code, denied)
            self.assertIn(client.delete('/conversations/' + cid, headers=MUTATION).status_code, denied)
            self.assertIn(self.submit('Уточнение', cid, client).status_code, denied)
            self.assertIn(client.get('/ask/jobs/' + first['job_id']).status_code, (401, 404))
        self.assertEqual(len(self.client.get('/conversations').json()['conversations']), 1)
        self.assertEqual(other.get('/conversations').json()['conversations'], [])
        own = self.client.get('/conversations/' + cid)
        self.assertEqual(own.headers.get('cache-control'), 'no-store')
        self.assertEqual(own.json()['turns'][0]['question'], 'Что такое чернозём?')

    def test_saved_chat_and_photo_memory_reload_after_ram_reset(self):
        self.register()
        output = io.BytesIO()
        Image.new('RGB', (12, 12)).save(output, format='PNG')
        first = self.new_chat('Что на фото?', image=base64.b64encode(output.getvalue()).decode())
        cid = first['conversation_id']
        api.app.state.conversations.clear()
        restored = self.client.get('/conversations/' + cid)
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertTrue(restored.json()['persisted'])
        self.assertTrue(restored.json()['turns'][0]['has_image'])
        response = self.submit('А почему верхний слой тёмный?', cid)
        self.assertEqual(response.status_code, 202, response.text)
        self.done(response.json()['job_id'])
        memory = self.calls[-1][2]
        self.assertIn('Тёмный горизонт', memory['image_description'])
        self.assertEqual(memory['history'][0]['question'], 'Что на фото?')
        self.assertIsNone(self.calls[-1][1])

    def test_deleting_chat_does_not_delete_other_saved_chat(self):
        self.register()
        first = self.new_chat('Первая почва')
        second = self.new_chat('Вторая почва')
        response = self.client.delete('/conversations/' + first['conversation_id'], headers=MUTATION)
        self.assertEqual(response.status_code, 204, response.text)
        self.assertEqual(self.client.get('/conversations/' + first['conversation_id']).status_code, 404)
        remaining = self.client.get('/conversations').json()['conversations']
        self.assertEqual([x['conversation_id'] for x in remaining], [second['conversation_id']])
        with db.get_conn() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM messages WHERE conversation_id=?', (first['conversation_id'],)).fetchone()[0], 0)

    def test_deleting_active_chat_cannot_bypass_account_inference_limit(self):
        self.register()
        self.release.clear()
        self.started.clear()
        first = self.submit('Удаляемый вопрос').json()
        self.assertTrue(self.started.wait(5))
        blocked = self.submit('Новая беседа во время обработки')
        self.assertEqual(blocked.status_code, 429, blocked.text)
        self.assertIn('retry-after', blocked.headers)
        self.assertEqual(self.client.delete('/conversations/' + first['conversation_id'], headers=MUTATION).status_code, 204)
        self.assertEqual(self.submit('Обход через удаление').status_code, 429)
        self.release.set()
        for _ in range(500):
            if self.inflight == 0:
                break
            time.sleep(.01)
        time.sleep(.03)  # allow the FIFO worker to finalize its already released job
        second = self.new_chat('После завершения')
        self.assertNotEqual(first['conversation_id'], second['conversation_id'])
        self.assertEqual(self.max_inflight, 1)
        self.assertEqual(self.client.get('/conversations/' + first['conversation_id']).status_code, 404)
        self.assertEqual(self.client.get('/ask/jobs/' + first['job_id']).status_code, 404)

    def test_new_chat_creation_rate_limit_is_enforced_by_server(self):
        self.register()
        for i in range(5):
            self.new_chat('Почва ' + str(i))
        blocked = self.submit('Шестая почва')
        self.assertEqual(blocked.status_code, 429, blocked.text)
        self.assertIn('retry-after', blocked.headers)
        self.assertEqual(len(self.client.get('/conversations').json()['conversations']), 5)

    def test_delete_account_during_inference_cannot_recreate_messages(self):
        response = self.register()
        uid = response.json()['user']['id']
        self.release.clear()
        self.started.clear()
        first = self.submit('Вопрос удаляемого аккаунта').json()
        self.assertTrue(self.started.wait(5))
        deleted = self.client.request('DELETE', '/account', json={'password': PASSWORD}, headers=MUTATION)
        self.assertEqual(deleted.status_code, 204, deleted.text)
        self.release.set()
        # A second account's completed job proves the worker finished the old job.
        other = self.other_client()
        self.register(other, 'survivor@example.invalid')
        self.new_chat('Контрольный вопрос', other)
        with db.get_conn() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM users WHERE id=?', (uid,)).fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM messages WHERE user_id=? OR conversation_id=?',
                                         (uid, first['conversation_id'])).fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM conversations WHERE id=?',
                                         (first['conversation_id'],)).fetchone()[0], 0)
        self.assertIn(self.client.get('/ask/jobs/' + first['job_id']).status_code, (401, 404))

    def test_delete_account_checks_password_and_removes_all_linked_data(self):
        response = self.register()
        uid = response.json()['user']['id']
        self.new_chat()
        bad = self.client.request('DELETE', '/account', json={'password': 'wrong'}, headers=MUTATION)
        self.assertIn(bad.status_code, (400, 403), bad.text)
        self.assertEqual(self.client.get('/me').status_code, 200)
        deleted = self.client.request('DELETE', '/account', json={'password': PASSWORD}, headers=MUTATION)
        self.assertEqual(deleted.status_code, 204, deleted.text)
        self.assertEqual(self.client.get('/me').status_code, 401)
        with db.get_conn() as conn:
            for table, column in [('users', 'id'), ('sessions', 'user_id'), ('messages', 'user_id'), ('conversations', 'user_id')]:
                self.assertEqual(conn.execute(f'SELECT COUNT(*) FROM {table} WHERE {column}=?', (uid,)).fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
