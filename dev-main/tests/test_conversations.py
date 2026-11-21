import asyncio
import copy
import base64
import io
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

import api
import db
from conversations import ConversationStore, ConversationExpired, ConversationBusy, ConversationCapacity


class ConversationStoreTests(unittest.TestCase):
    def test_photo_cache_has_fixed_expiry_even_when_chat_is_active(self):
        now = [0]
        store = ConversationStore(clock=lambda: now[0], image_ttl=10)
        cid, _ = store.reserve()
        store.complete(cid, 'photo', 'answer', has_image=True,
                       image_description='description', image_bytes=b'pixels')
        store.release(cid)
        now[0] = 9
        _, memory = store.reserve(cid)
        self.assertEqual(memory['remembered_image_bytes'], b'pixels')
        store.complete(cid, 'followup', 'answer')
        store.release(cid)
        self.assertNotIn('remembered_image_bytes', store.snapshot(cid))
        self.assertNotIn('image_bytes', store.snapshot(cid))
        now[0] = 10
        _, memory = store.reserve(cid)
        self.assertNotIn('remembered_image_bytes', memory)
        self.assertEqual(memory['image_description'], 'description')

    def test_photo_budget_evicts_old_pixels_and_new_site_removes_latest_pixels(self):
        store = ConversationStore(max_image_bytes=6)
        first, _ = store.reserve()
        store.complete(first, 'photo', 'a', has_image=True, image_bytes=b'1234')
        store.release(first)
        second, _ = store.reserve()
        store.complete(second, 'photo', 'a', has_image=True, image_bytes=b'5678')
        store.release(second)
        self.assertNotIn('remembered_image_bytes', store.reserve(first)[1])
        self.assertEqual(store.reserve(second)[1]['remembered_image_bytes'], b'5678')
        store.complete(second, 'new site', 'a', clear_photo=True)
        store.release(second)
        self.assertNotIn('remembered_image_bytes', store.reserve(second)[1])

    def test_photo_cache_is_owner_scoped_and_not_recovered_from_sqlite(self):
        store = ConversationStore()
        cid, _ = store.reserve(owner_id=1)
        store.complete(cid, 'photo', 'a', has_image=True, image_bytes=b'private-pixels')
        store.release(cid)
        with self.assertRaises(ConversationExpired):
            store.reserve(cid, owner_id=2)
        store.delete_owner(1)
        with patch.object(db, 'get_conversation_memory', return_value={
                'history': [], 'image_description': 'saved description'}) as load:
            _, memory = store.reserve(cid, owner_id=1)
            load.assert_called_once_with(cid, 1)
            self.assertNotIn('remembered_image_bytes', memory)

    def test_isolation_bounded_history_and_photo_replacement(self):
        store = ConversationStore(max_turns=2, max_chars=120)
        first, _ = store.reserve()
        store.complete(first, "q", "a", has_image=True, image_description="first photo")
        store.release(first)
        other, memory = store.reserve()
        self.assertEqual(memory, {"history": [], "image_description": ""})
        store.release(other)
        for i in range(5):
            _, memory = store.reserve(first)
            self.assertEqual(memory["image_description"], "first photo")
            store.complete(first, str(i), "answer")
            store.release(first)
        turns = store.snapshot(first)["turns"]
        self.assertEqual([t["question"] for t in turns], ["3", "4"])
        store.reserve(first)
        store.complete(first, "new photo", "a", has_image=True, image_description="replacement")
        store.release(first)
        self.assertEqual(store.reserve(first)[1]["image_description"], "replacement")

    def test_expiry_busy_capacity_and_deletion_during_inference(self):
        now = [0]
        store = ConversationStore(ttl=10, max_conversations=1, clock=lambda: now[0])
        cid, _ = store.reserve()
        with self.assertRaises(ConversationBusy): store.reserve(cid)
        with self.assertRaises(ConversationCapacity): store.reserve()
        now[0] = 11
        self.assertEqual(store.snapshot(cid)["conversation_id"], cid)  # active work protected
        store.release(cid)
        now[0] = 22
        with self.assertRaises(ConversationExpired): store.snapshot(cid)
        new, _ = store.reserve()
        store.delete(new)
        store.complete(new, "late", "late answer")
        store.release(new)
        with self.assertRaises(ConversationExpired): store.snapshot(new)

    def test_snapshot_is_a_copy_and_reads_do_not_extend_ttl(self):
        now = [0]
        store = ConversationStore(ttl=10, clock=lambda: now[0])
        cid, _ = store.reserve()
        store.complete(cid, "original", "answer")
        store.release(cid)
        state = store.snapshot(cid)
        state["turns"][0]["question"] = "mutated"
        self.assertEqual(store.snapshot(cid)["turns"][0]["question"], "original")
        now[0] = 11
        with self.assertRaises(ConversationExpired): store.snapshot(cid)


class ConversationAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(db, "DB_FILE", str(Path(self.temp.name) / "test.db")),
                        patch.object(api, "preload_reranker")]
        for p in self.patches: p.start()
        self.release = threading.Event(); self.release.set()
        self.started = threading.Event()
        self.inputs = []
        def model(text, image, memory=None):
            self.inputs.append((text, image, copy.deepcopy(memory)))
            self.started.set()
            if not self.release.wait(5): raise RuntimeError("test stalled")
            if text == "fail": raise RuntimeError("model failed")
            return {"answer": "Ответ: " + text, "image_description": "Тёмный слой" if image else ""}
        self.patches.append(patch.object(api, "run_agent", side_effect=model)); self.patches[-1].start()
        self.client = TestClient(api.app); self.client.__enter__()

    def tearDown(self):
        self.release.set(); self.client.__exit__(None, None, None)
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def submit(self, text, cid=None, image=None):
        body = {"text": text}
        if cid: body["conversation_id"] = cid
        if image: body["image"] = image
        return self.client.post("/ask/jobs", json=body)

    def done(self, jid):
        for _ in range(500):
            result = self.client.get('/ask/jobs/' + jid).json()
            if result.get("status") in {"done", "failed"}: return result
            time.sleep(.01)
        self.fail("timeout")

    def test_successful_turns_are_remembered_and_guest_transcript_not_persisted(self):
        first = self.submit("Расскажи про чернозём").json(); self.done(first['job_id'])
        cid = first['conversation_id']
        next_ = self.submit("А где он встречается?", cid).json(); self.done(next_['job_id'])
        self.assertEqual(self.inputs[-1][2]['history'][0]['question'], "Расскажи про чернозём")
        independent = self.submit("Новая тема").json(); self.done(independent['job_id'])
        self.assertIsNone(self.inputs[-1][2])
        snapshot = self.client.get('/conversations/' + cid)
        self.assertEqual(snapshot.headers['cache-control'], 'no-store')
        self.assertEqual(len(snapshot.json()['turns']), 2)
        with db.get_conn() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 0)

    def test_failure_and_conflict_do_not_add_turns_or_leave_busy(self):
        first = self.submit("first").json(); self.done(first['job_id']); cid = first['conversation_id']
        self.release.clear(); self.started.clear()
        failed = self.submit("fail", cid).json(); self.assertTrue(self.started.wait(5))
        self.assertEqual(self.submit("parallel", cid).status_code, 429)
        self.assertEqual(self.client.get('/conversations/' + cid).json()['active_job_id'], failed['job_id'])
        self.release.set(); self.assertEqual(self.done(failed['job_id'])['status'], 'failed')
        retry = self.submit("retry", cid); self.assertEqual(retry.status_code, 202)
        self.done(retry.json()['job_id'])
        self.assertEqual([t['question'] for t in self.client.get('/conversations/' + cid).json()['turns']], ['first', 'retry'])

    def test_deleting_active_conversation_drops_late_result_and_memory(self):
        self.release.clear()
        first = self.submit("delete me").json(); cid = first['conversation_id']
        self.assertTrue(self.started.wait(5))
        self.assertEqual(self.client.delete('/conversations/' + cid).status_code, 204)
        self.release.set()
        self.assertEqual(self.client.get('/conversations/' + cid).status_code, 410)
        self.assertEqual(self.submit("followup", cid).status_code, 410)
        self.assertEqual(self.client.get('/ask/jobs/' + first['job_id']).status_code, 404)
        second = self.submit("new conversation").json(); self.done(second['job_id'])
        self.assertNotEqual(cid, second['conversation_id'])
        self.assertIsNone(self.inputs[-1][2])

    def test_photo_pixels_are_internal_only_and_available_on_text_followup(self):
        stream = io.BytesIO(); Image.new('RGB', (8, 8)).save(stream, format='PNG')
        first = self.submit("Фото", image=base64.b64encode(stream.getvalue()).decode()).json(); self.done(first['job_id'])
        followup = self.submit("Что сверху?", first['conversation_id']).json(); self.done(followup['job_id'])
        self.assertIsNone(self.inputs[-1][1])
        self.assertEqual(self.inputs[-1][2]['image_description'], 'Тёмный слой')
        self.assertTrue(self.inputs[-1][2]['remembered_image_bytes'].startswith(b'\xff\xd8'))
        saved = self.client.get('/conversations/' + first['conversation_id']).json()
        self.assertTrue(saved['turns'][0]['has_image'])
        self.assertNotIn('image_description', saved)
        self.assertNotIn('remembered_image_bytes', saved)
        self.assertNotIn('image_bytes', saved)

    def test_expired_and_oversized_requests_never_reach_model(self):
        self.assertEqual(self.submit('q', 'x' * 43).status_code, 410)
        self.assertEqual(self.submit('q', 'x' * 10000).status_code, 422)
        self.assertEqual(self.submit('x' * 4001).status_code, 422)
        self.assertEqual(self.inputs, [])

    def test_missing_admin_key_fails_closed(self):
        with patch.object(api, 'ADMIN_KEY', None):
            self.assertEqual(self.client.get('/knowledge').status_code, 403)


if __name__ == '__main__': unittest.main()
