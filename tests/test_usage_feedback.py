import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
import api
import db
import usage

class UsageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.patch=patch.object(db,'DB_FILE',str(Path(self.temp.name)/'app.db'));self.patch.start();db.init_db()
        with db.get_conn() as c:
            self.uid=c.execute("INSERT INTO users(name,email,password_hash) VALUES ('Tester','test@invalid','unused')").lastrowid
        self.owner='user:'+str(self.uid)
        self.cid='c'*43
        db.create_conversation(self.cid,self.uid,'Почва')
        db.save_conversation_turn(self.cid,self.uid,'Вопрос','Ответ')
    def tearDown(self):
        self.patch.stop();self.temp.cleanup()
    def test_concurrent_quota_refund_and_next_day(self):
        def take(_):
            try:return usage.reserve(self.owner,self.uid)
            except ValueError:return None
        with ThreadPoolExecutor(max_workers=8) as pool:tokens=list(pool.map(take,range(12)))
        self.assertEqual(len([t for t in tokens if t]),5)
        token=next(t for t in tokens if t);usage.finish(token,False)
        self.assertEqual(usage.status(self.owner,self.uid)['remaining'],1)
        with patch.object(usage,'day',return_value='2099-01-01'):
            self.assertEqual(usage.status(self.owner,self.uid)['remaining'],5)
    def test_low_rating_gets_bonus_only_once_and_delete_cascades(self):
        usage.submit_feedback(self.uid,self.cid,1,'Ответ был неполным, хотелось бы ссылки.')
        self.assertEqual(usage.status(self.owner,self.uid)['remaining'],10)
        with self.assertRaises(ValueError):usage.submit_feedback(self.uid,self.cid,5,'Теперь всё замечательно!')
        with patch.object(usage,'day',return_value='2099-01-01'):
            self.assertEqual(usage.status(self.owner,self.uid)['remaining'],5)
            self.assertFalse(usage.status(self.owner,self.uid)['bonus_available'])
        usage.reserve(self.owner,self.uid);db.delete_user(self.uid)
        with db.get_conn() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM feedback').fetchone()[0],0)
            self.assertEqual(c.execute('SELECT count(*) FROM request_usage').fetchone()[0],0)
    def test_feedback_requires_owned_answer_and_real_comment(self):
        for cid,comment in [('x'*43,'Достаточно длинный отзыв'),(self.cid,' '*20)]:
            with self.assertRaises(ValueError):usage.submit_feedback(self.uid,cid,3,comment)
        other=db.create_user('Other','other@invalid','A-unique-phrase-123')
        with self.assertRaises(ValueError):usage.submit_feedback(other,self.cid,3,'Достаточно длинный отзыв')
    def test_admin_and_restart_recovery(self):
        pending=usage.reserve(self.owner,self.uid)
        complete=usage.reserve(self.owner,self.uid);usage.finish(complete,True)
        usage.recover();self.assertEqual(usage.status(self.owner,self.uid)['remaining'],4)
        with db.get_conn() as c:c.execute("UPDATE users SET role='admin' WHERE id=?",(self.uid,))
        for _ in range(10):self.assertIsNone(usage.reserve(self.owner,self.uid))
        self.assertTrue(usage.status(self.owner,self.uid)['admin'])
    def test_api_authorization_and_sixth_question(self):
        with patch.object(api,'preload_reranker'),patch.object(api,'run_agent',return_value={'answer':'Ответ'}),TestClient(api.app) as client:
            token='t'*64
            with db.get_conn() as c:c.execute("INSERT INTO sessions(token,user_id,expires_at) VALUES (?,?,datetime('now','+1 day'))",(token,self.uid))
            headers={'x-token':token}
            self.assertEqual(client.get('/admin/feedback',headers=headers).status_code,403)
            self.assertEqual(client.post('/feedback',json={'conversation_id':self.cid,'stars':4,'comment':'Мне понравились подробные объяснения.'}).status_code,401)
            for _ in range(5):usage.finish(usage.reserve(self.owner,self.uid),True)
            r=client.post('/ask/jobs',headers=headers,json={'text':'Почва','conversation_id':self.cid})
            self.assertEqual(r.status_code,429,r.text)
            r=client.post('/feedback',headers=headers,json={'conversation_id':self.cid,'stars':1,'comment':'Не хватило подробного объяснения.'})
            self.assertEqual(r.status_code,201,r.text);self.assertEqual(r.json()['remaining'],5)
            self.assertEqual(client.get('/account/usage',headers=headers).headers['cache-control'],'no-store')
            with db.get_conn() as c:c.execute("UPDATE users SET role='admin' WHERE id=?",(self.uid,))
            self.assertEqual(client.get('/admin/feedback',headers=headers).json()['feedback'][0]['stars'],1)

    def test_model_failure_returns_reservation_and_guest_limit_applies(self):
        import time
        with patch.object(api,'preload_reranker'),patch.object(api,'run_agent',side_effect=RuntimeError('model unavailable')),TestClient(api.app) as client:
            r=client.post('/ask/jobs',json={'text':'Расскажите о почве'})
            self.assertEqual(r.status_code,202,r.text)
            for _ in range(100):
                state=client.get('/ask/jobs/'+r.json()['job_id']).json()
                if state['status']=='failed':break
                time.sleep(.01)
            self.assertEqual(state['status'],'failed')
            self.assertEqual(client.get('/account/usage').json()['remaining'],5)
            # Exercise the public endpoint after five successful guest reservations.
            from account_security import GUEST_COOKIE
            cookie=client.cookies.get(GUEST_COOKIE)
            owner='guest:'+cookie.split('.')[0]
            for _ in range(5):usage.finish(usage.reserve(owner,None),True)
            self.assertEqual(client.post('/ask/jobs',json={'text':'Ещё вопрос','conversation_id':r.json()['conversation_id']}).status_code,429)
