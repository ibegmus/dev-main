import re
import secrets
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import db


class PersistentDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = patch.object(db, 'DB_FILE', str(Path(self.temp.name) / 'accounts.db'))
        self.patch.start()
        db.init_db()
        self.user = db.create_user('First', 'First@example.test', 'password123', '2026-11-04.accounts')
        self.other = db.create_user('Other', 'other@example.test', 'password456')

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def create_chat(self, user=None):
        cid = secrets.token_urlsafe(32)
        self.assertTrue(db.create_conversation(cid, self.user if user is None else user, 'What soil?'))
        return cid

    def test_sessions_expire_revoke_and_account_terms_are_recorded(self):
        token = db.login(' FIRST@EXAMPLE.TEST ', 'password123')
        self.assertEqual(db.get_user_by_token(token)['id'], self.user)
        self.assertTrue(db.verify_password(self.user, 'password123'))
        self.assertFalse(db.verify_password(self.user, 'wrong'))
        with db.get_conn() as conn:
            user = conn.execute('SELECT * FROM users WHERE id = ?', (self.user,)).fetchone()
            self.assertEqual(user['terms_version'], '2026-11-04.accounts')
            self.assertTrue(user['terms_accepted_at'])
            self.assertNotEqual(user['password_hash'], 'password123')
            conn.execute("UPDATE sessions SET expires_at = datetime('now', '-1 second') WHERE token = ?", (token,))
        self.assertIsNone(db.get_user_by_token(token))
        token2 = db.login('first@example.test', 'password123')
        db.revoke_session(token2)
        self.assertIsNone(db.get_user_by_token(token2))

    def test_owner_is_required_for_all_chat_operations(self):
        cid = self.create_chat()
        self.assertFalse(db.save_conversation_turn(cid, self.other, 'q', 'a'))
        self.assertIsNone(db.get_conversation(cid, self.other))
        self.assertIsNone(db.get_conversation_turns(cid, self.other))
        self.assertIsNone(db.get_conversation_memory(cid, self.other))
        self.assertFalse(db.delete_conversation(cid, self.other))
        self.assertEqual(db.list_conversations(self.other), [])
        self.assertTrue(db.get_conversation(cid, self.user))

    def test_persistence_pagination_and_bounded_model_memory(self):
        cid = self.create_chat()
        for n in range(105):
            self.assertTrue(db.save_conversation_turn(cid, self.user, f'question {n}', f'answer {n}'))
        db.init_db()
        latest = db.get_conversation_turns(cid, self.user)
        self.assertEqual(len(latest['turns']), 100)
        self.assertTrue(latest['has_more'])
        self.assertEqual(latest['turns'][0]['question'], 'question 5')
        previous = db.get_conversation_turns(cid, self.user, before=latest['next_before'])
        self.assertEqual(len(previous['turns']), 5)
        self.assertFalse(previous['has_more'])
        memory = db.get_conversation_memory(cid, self.user)
        self.assertEqual(len(memory['history']), 8)
        self.assertEqual(memory['history'][-1]['answer'], 'answer 104')

    def test_photo_description_is_replaced_and_retained_without_raw_photo(self):
        cid = self.create_chat()
        db.save_conversation_turn(cid, self.user, 'photo', 'answer', has_image=True, image_description='dark layer')
        db.save_conversation_turn(cid, self.user, 'followup', 'answer')
        self.assertEqual(db.get_conversation_memory(cid, self.user)['image_description'], 'dark layer')
        history = db.get_history(self.user)
        self.assertIn('[с фото]', history[1]['question'])
        db.save_conversation_turn(cid, self.user, 'new photo', 'answer', has_image=True, image_description='red layer')
        self.assertEqual(db.get_conversation_memory(cid, self.user)['image_description'], 'red layer')
        with db.get_conn() as conn:
            self.assertNotIn('image', [r[1] for r in conn.execute('PRAGMA table_info(messages)')])

    def test_soil_case_survives_reopen_and_is_owner_scoped(self):
        cid = self.create_chat()
        case = {'version': 1, 'user_notes': ['Суглинок рядом с Пермью'],
                'photos': ['Виден светлый слой']}
        self.assertTrue(db.save_conversation_turn(cid, self.user, 'Определи почву', 'Нужно фото',
                                                  soil_case=case))
        db.init_db()
        self.assertEqual(db.get_conversation_memory(cid, self.user)['soil_case'], case)
        self.assertIsNone(db.get_conversation_memory(cid, self.other))
        db.save_conversation_turn(cid, self.user, 'Другой участок', 'Начнём заново',
                                  soil_case={'version': 1, 'user_notes': ['Другой участок'], 'photos': []},
                                  clear_photo=True)
        memory = db.get_conversation_memory(cid, self.user)
        self.assertEqual(memory['soil_case']['photos'], [])
        self.assertEqual(memory['image_description'], '')

    def test_structured_interview_survives_database_reopen(self):
        from soil_case import record_case
        from soil_interview import reply
        cid=self.create_chat()
        case=record_case({},'Определи почву')
        answer,case=reply(case,'Определи почву')
        db.save_conversation_turn(cid,self.user,'Определи почву',answer,soil_case=case)
        db.init_db()
        loaded=db.get_conversation_memory(cid,self.user)['soil_case']
        self.assertEqual(loaded['pending_field'],'location')
        updated=record_case(loaded,'Не знаю')
        answer,updated=reply(updated,'Не знаю')
        self.assertEqual(updated['fields']['location']['status'],'unknown')
        self.assertNotIn('В каком регионе',answer)

    def test_reference_hypothesis_can_resume_from_saved_observations(self):
        from soil_case import record_case
        from soil_interview import reply
        cid=self.create_chat()
        question='Свердловская область. Внутри профиля светлый слой'
        case=record_case({},question,updates=[
            {'field':'location','status':'known','quote':'Свердловская область'},
            {'field':'profile','status':'known','quote':'Внутри профиля светлый слой'}])
        answer,case=reply(case,question)
        self.assertEqual(case['pending_field'],'lower_layer')
        db.save_conversation_turn(cid,self.user,question,answer,soil_case=case)
        db.init_db()
        loaded=db.get_conversation_memory(cid,self.user)['soil_case']
        updated=record_case(loaded,'Рыжий песок')
        answer,updated=reply(updated,'Подведите итог')
        self.assertNotIn('\nПодзолы\n',answer)
        self.assertIn('Рыжевато-бурый материал ниже тоже учтён',answer)
        self.assertIn('https://soils.narod.ru/taxon/type/podz07.html',answer)
        self.assertEqual(updated['fields']['lower_layer']['quote'],'Рыжий песок')
        self.assertIsNone(db.get_conversation_memory(cid,self.other))

    def test_pending_photo_confirmation_survives_database_reopen(self):
        from soil_case import record_case
        from soil_interview import reply
        from soil_photo import validated_observation, awaiting_confirmation, confirmed_features
        cid=self.create_chat()
        observation=validated_observation({'photo_id':'a'*64,'scene':'profile','quality':'usable',
            'color':'natural','features':['pale_horizon','rust_lower']})
        case=record_case({},'Свердловская область','Светлый слой',new_photo=True,
            photo_observation=observation,updates=[
                {'field':'location','status':'known','quote':'Свердловская область'}])
        answer,case=reply(case,'Определи почву')
        db.save_conversation_turn(cid,self.user,'Определи почву',answer,soil_case=case)
        db.init_db()
        loaded=db.get_conversation_memory(cid,self.user)['soil_case']
        self.assertTrue(awaiting_confirmation(loaded))
        updated=record_case(loaded,'Да')
        self.assertEqual(confirmed_features(updated)[0]['feature'],'pale_horizon')
        self.assertNotIn('image_bytes',str(updated))
        self.assertIsNone(db.get_conversation_memory(cid,self.other))

    def test_semantic_observations_survive_reopen_without_parsing_old_assistant_answers(self):
        from soil_case import record_case
        from soil_knowledge import evaluate
        cid=self.create_chat()
        statement='Сверху тёмная полоска, под ней светлая, а ниже рыжеватая земля.'
        case=record_case({},statement)
        for _ in range(12):case=record_case(case,'Продолжим')
        db.save_conversation_turn(cid,self.user,statement,'Это не доказанный подзол',soil_case=case)
        db.init_db()
        loaded=db.get_conversation_memory(cid,self.user)['soil_case']
        self.assertEqual(loaded['version'],4)
        self.assertIn('pale_horizon',evaluate(loaded)['evidence']['present'])
        self.assertIn('rust_lower',evaluate(loaded)['evidence']['present'])
        self.assertEqual(evaluate(loaded)['candidates'],[])
        reset=record_case(loaded,'Другой участок',reset=True)
        self.assertEqual(evaluate(reset)['evidence']['present'],{})

    def test_delete_chat_and_user_prevent_late_result_resurrection(self):
        cid = self.create_chat()
        db.save_conversation_turn(cid, self.user, 'q', 'a')
        self.assertTrue(db.delete_conversation(cid, self.user))
        self.assertFalse(db.save_conversation_turn(cid, self.user, 'late', 'late answer'))
        cid = self.create_chat()
        db.save_conversation_turn(cid, self.user, 'q2', 'a2')
        token = db.login('first@example.test', 'password123')
        # Also remove ungrouped legacy history rather than leaving SET NULL text.
        db.save_message(self.user, 'legacy', 'legacy answer', None, 1)
        self.assertTrue(db.delete_user(self.user))
        self.assertFalse(db.save_conversation_turn(cid, self.user, 'late', 'late answer'))
        self.assertIsNone(db.get_user_by_token(token))
        with db.get_conn() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM conversations').fetchone()[0], 0)
        self.assertIsNotNone(db.get_user(self.other))

    def test_conversation_capacity_does_not_remove_existing(self):
        with patch.object(db, 'MAX_SAVED_CONVERSATIONS', 2):
            first = self.create_chat()
            self.create_chat()
            self.assertFalse(db.create_conversation(secrets.token_urlsafe(32), self.user, 'third'))
            self.assertEqual(len(db.list_conversations(self.user)), 2)
            db.delete_conversation(first, self.user)
            self.create_chat()

    def test_memory_character_budget_keeps_latest_exchange(self):
        cid = self.create_chat()
        for i in range(10):
            db.save_conversation_turn(cid, self.user, str(i) + 'q' * 3999, 'a' * 6000)
        turns = db.get_conversation_memory(cid, self.user)['history']
        self.assertEqual(turns[-1]['question'][0], '9')
        self.assertLessEqual(sum(len(t['question']) + len(t['answer']) for t in turns), 24000)


class OldDatabaseMigrationTests(unittest.TestCase):
    def test_legacy_rows_are_preserved_grouped_once_and_sessions_expire(self):
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / 'old.db')
            conn = sqlite3.connect(path)
            conn.executescript('''
                CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, password_hash TEXT, created_at TEXT);
                CREATE TABLE sessions(token TEXT PRIMARY KEY, user_id INTEGER REFERENCES users(id) ON DELETE CASCADE, created_at TEXT);
                CREATE TABLE messages(id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    question TEXT, answer TEXT, intent TEXT, seconds REAL, created_at TEXT);
                INSERT INTO users VALUES (1, 'Legacy', 'legacy@example.test', 'hash', '2020-01-01');
                INSERT INTO sessions VALUES ('old-token', 1, '2020-01-01');
                INSERT INTO messages VALUES (1, 1, 'old q [с фото]', 'old a', 'science', 2, '2020-01-01');
                INSERT INTO messages VALUES (2, NULL, 'guest q', 'guest a', NULL, 1, '2020-01-01');
            ''')
            conn.close()
            with patch.object(db, 'DB_FILE', path):
                db.init_db()
                chats = db.list_conversations(1)
                self.assertEqual(len(chats), 1)
                cid = chats[0]['conversation_id']
                self.assertRegex(cid, r'^[A-Za-z0-9_-]{43}$')
                self.assertEqual(db.get_conversation_turns(cid, 1)['turns'][0]['answer'], 'old a')
                self.assertTrue(db.get_conversation_turns(cid, 1)['turns'][0]['has_image'])
                self.assertIsNone(db.get_user_by_token('old-token'))
                db.init_db()
                self.assertEqual(db.list_conversations(1)[0]['conversation_id'], cid)
                with db.get_conn() as conn:
                    self.assertEqual(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 2)
                    self.assertIsNone(conn.execute('SELECT terms_version FROM users WHERE id = 1').fetchone()[0])


if __name__ == '__main__':
    unittest.main()
