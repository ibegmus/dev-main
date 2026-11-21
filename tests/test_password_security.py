import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
import db
from password_security import hash_password, password_matches, password_problem, ITERATIONS


class PasswordTests(unittest.TestCase):
    def test_policy_accepts_phrases_and_rejects_common_or_repeated_passwords(self):
        for value in ['12345678', 'Password123456!', 'Qwerty123456', 'aaaaaaaaaaaa', 'пароль123456789']:
            self.assertIsNotNone(password_problem(value), value)
        self.assertIsNone(password_problem('Три разных слова у почвенного музея'))
        self.assertIsNotNone(password_problem('visitor-name@example.invalid', 'visitor-name@example.invalid'))

    def test_hashes_are_salted_and_legacy_hashes_still_verify(self):
        password = 'A-long-unique-phrase-95'
        a, b = hash_password(password), hash_password(password)
        self.assertNotEqual(a, b)
        self.assertIn(str(ITERATIONS), a)
        self.assertTrue(password_matches(a, password))
        self.assertFalse(password_matches(a, 'wrong'))
        salt = '00' * 16
        legacy = salt + '$' + db._hash_password(password, salt)
        self.assertTrue(password_matches(legacy, password))
        self.assertFalse(password_matches('pbkdf2_sha256$999999999$'+salt+'$'+'0'*64, password))


class PasswordAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(db, 'DB_FILE', str(Path(self.temp.name)/'test.db')), patch.object(api, 'preload_reranker')]
        for p in self.patches: p.start()
        self.client = TestClient(api.app, base_url='http://localhost');self.client.__enter__()
        self.headers={'X-TerraMind-Request':'1'}
        self.password='Original-long-password-17'

    def tearDown(self):
        self.client.__exit__(None,None,None)
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def register(self, browser=False):
        h={**self.headers, **({'Origin':'http://localhost','Sec-Fetch-Site':'same-origin'} if browser else {})}
        return self.client.post('/register',json={'name':'Test','email':'test@example.invalid','password':self.password,'terms_accepted':True},headers=h)

    def test_password_change_revokes_old_sessions_and_preserves_chats(self):
        first = self.register().json(); uid=first['user']['id'];oldtoken=first['token']
        second = db.login('test@example.invalid',self.password)
        db.create_conversation('c'*43,uid,'Сохранённая беседа')
        response=self.client.post('/account/password',json={'current_password':self.password,'new_password':'Different-unique-phrase-92'},headers=self.headers)
        self.assertEqual(response.status_code,200,response.text)
        self.assertIsNone(db.get_user_by_token(oldtoken));self.assertIsNone(db.get_user_by_token(second))
        self.assertEqual(self.client.get('/me').status_code,200)
        self.assertEqual(len(db.list_conversations(uid)),1)
        self.assertIsNone(db.login('test@example.invalid',self.password))
        self.assertIsNotNone(db.login('test@example.invalid','Different-unique-phrase-92'))

    def test_rejected_changes_keep_password_and_require_cookie_csrf_header(self):
        self.register()
        for current,new in [('wrong','A-different-password-17'),(self.password,'Password123456'),(self.password,self.password)]:
            r=self.client.post('/account/password',json={'current_password':current,'new_password':new},headers=self.headers)
            self.assertIn(r.status_code,(400,403))
        self.assertEqual(self.client.post('/account/password',json={'current_password':self.password,'new_password':'Another-valid-phrase-57'}).status_code,403)
        self.assertTrue(db.verify_password(self.client.get('/me').json()['user']['id'],self.password))

    def test_browser_gets_cookie_not_json_token_and_errors_do_not_echo_password(self):
        response=self.register(browser=True)
        self.assertEqual(response.status_code,201,response.text)
        self.assertNotIn('token',response.json())
        self.assertIn('httponly',response.headers['set-cookie'].lower())
        secret='SECRET-'*40
        r=self.client.post('/account/password',json={'current_password':secret,'new_password':'Unique-phrase-95'},headers=self.headers)
        self.assertEqual(r.status_code,422)
        self.assertNotIn(secret,r.text)
        self.assertTrue(all('input' not in x for x in r.json()['detail']))
        self.assertEqual(r.headers['cache-control'],'no-store')

    def test_login_upgrades_old_hash_and_cannot_reissue_session_after_password_changed(self):
        uid=db.create_user('Legacy','legacy@example.invalid',self.password)
        salt='01'*16;old=salt+'$'+db._hash_password(self.password,salt)
        with db.get_conn() as conn:conn.execute('UPDATE users SET password_hash=? WHERE id=?',(old,uid))
        self.assertIsNotNone(db.login('legacy@example.invalid',self.password))
        with db.get_conn() as conn:self.assertTrue(conn.execute('SELECT password_hash FROM users WHERE id=?',(uid,)).fetchone()[0].startswith('pbkdf2_sha256$600000$'))
        original=db._password_matches
        def change_after_check(stored,pwd):
            good=original(stored,pwd)
            db.change_password(uid,self.password,'Replacement-valid-phrase-77')
            return good
        with patch.object(db,'_password_matches',side_effect=change_after_check):
            self.assertIsNone(db.login('legacy@example.invalid',self.password))

    def test_body_limits_reject_both_declared_and_streamed_oversize(self):
        response=self.client.post('/login',content=b'{}',headers={'Content-Type':'application/json','Content-Length':'99999999'})
        self.assertEqual(response.status_code,413)
        streamed=self.client.post('/register',content=iter([b'x'*9000,b'y'*9000]),headers={'Content-Type':'application/json'})
        self.assertEqual(streamed.status_code,413)
        self.assertEqual(streamed.headers['cache-control'],'no-store')

    def test_headers_and_same_origin_browser_protection(self):
        response=self.client.get('/')
        self.assertEqual(response.headers['x-frame-options'],'DENY')
        self.assertEqual(response.headers['x-content-type-options'],'nosniff')
        self.assertIn("frame-ancestors 'none'",response.headers['content-security-policy'])
        self.assertEqual(self.client.get('/docs').status_code,404)
        self.assertEqual(self.client.get('/openapi.json').status_code,200)
        r=self.client.post('/login',json={'email':'test@example.invalid','password':'pass'},headers={'Origin':'https://foreign.example'})
        self.assertEqual(r.status_code,403)
        self.assertNotIn('access-control-allow-origin',r.headers)


if __name__=='__main__':unittest.main()
