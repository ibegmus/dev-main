import asyncio
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from fastapi.testclient import TestClient

import api
import retrieval
from retrieval import HybridRetriever, get_reranker, preload_reranker
from request_queue import InferenceQueue


class Store:
    def __init__(self):
        self.docs = [Document(page_content="Образец собран на чайной плантации в Грузии в 1980 году.")]
        self.similarity_search = Mock(side_effect=lambda question, k: self.docs[:k])

    def get(self):
        return {"documents": [doc.page_content for doc in self.docs],
                "metadatas": [doc.metadata for doc in self.docs]}


class SearchCacheTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.model = SimpleNamespace(predict=Mock(side_effect=lambda pairs, **kwargs: [0.9] * len(pairs)))
        self.retriever = HybridRetriever(self.store, reranker_factory=lambda: self.model, cache_ttl=300)
        self.question = "Когда собран образец из Грузии?"

    def test_repeat_uses_identical_sources_without_repeating_embedding_or_ranking(self):
        first = self.retriever.search(self.question)
        expected_context = first["local_context"]
        # Callers must not be able to modify another request's cached result.
        first["sources"].clear()
        first["local_context"] = "Modified by caller"
        second = self.retriever.search(self.question)
        self.assertEqual(second["local_context"], expected_context)
        self.assertTrue(second["sources"])
        self.assertIn("1980", second["local_context"])
        self.store.similarity_search.assert_called_once()
        self.model.predict.assert_called_once()
        self.assertNotIn(self.question, repr(self.retriever._cache))

    def test_expiration_and_zero_ttl_repeat_real_search(self):
        clock = [0]
        with patch.object(retrieval, "time", SimpleNamespace(monotonic=lambda: clock[0])):
            self.retriever.search(self.question)
            clock[0] = 299
            self.retriever.search(self.question)
            self.assertEqual(self.model.predict.call_count, 1)
            clock[0] = 300
            self.retriever.search(self.question)
            self.assertEqual(self.model.predict.call_count, 2)
        uncached = HybridRetriever(self.store, reranker_factory=lambda: self.model, cache_ttl=0)
        uncached.search(self.question)
        uncached.search(self.question)
        self.assertEqual(self.model.predict.call_count, 4)

    def test_admin_refresh_discards_old_sources(self):
        self.retriever.search(self.question)
        self.store.docs = [Document(page_content="Новая подтверждённая запись об образце из Грузии.")]
        self.retriever.rebuild()
        result = self.retriever.search(self.question)
        self.assertIn("Новая подтверждённая запись", result["local_context"])
        self.assertNotIn("1980", result["local_context"])
        self.assertEqual(self.model.predict.call_count, 2)

    def test_search_finishing_after_refresh_cannot_populate_new_generation(self):
        def refresh_during_scoring(pairs, **kwargs):
            self.retriever.rebuild()
            return [0.9] * len(pairs)
        self.model.predict.side_effect = refresh_during_scoring
        self.retriever.search(self.question)
        self.assertFalse(self.retriever._cache)
        self.retriever.search(self.question)
        self.assertEqual(self.model.predict.call_count, 2)

    def test_lru_is_bounded_and_does_not_merge_different_questions(self):
        self.retriever.cache_size = 2
        for question in ["Грузия первый вопрос", "Грузия второй вопрос", "Грузия третий вопрос"]:
            self.retriever.search(question)
        self.assertEqual(len(self.retriever._cache), 2)
        self.retriever.search("Грузия второй вопрос")
        self.assertEqual(self.model.predict.call_count, 3)
        self.retriever.search("Грузия первый вопрос")
        self.assertEqual(self.model.predict.call_count, 4)

    def test_embedding_and_reranker_failures_are_retried(self):
        self.store.similarity_search.side_effect = ConnectionError("offline")
        with self.assertLogs("retrieval", level="ERROR"):
            self.retriever.search(self.question)
        self.assertFalse(self.retriever._cache)
        self.store.similarity_search.side_effect = lambda question, k: self.store.docs[:k]
        self.model.predict.side_effect = RuntimeError("temporary scorer failure")
        with self.assertLogs("retrieval", level="ERROR"):
            self.retriever.search(self.question)
        self.assertFalse(self.retriever._cache)
        self.model.predict.side_effect = lambda pairs, **kwargs: [0.9] * len(pairs)
        self.assertIn("1980", self.retriever.search(self.question)["local_context"])
        self.assertTrue(self.retriever._cache)
        self.assertEqual(self.store.similarity_search.call_count, 3)


class PreloadTests(unittest.TestCase):
    def setUp(self):
        get_reranker.cache_clear()
        self.addCleanup(get_reranker.cache_clear)
        environment = patch.dict(os.environ, {"RAG_PRELOAD": "1", "RAG_CPU_THREADS": ""})
        environment.start()
        self.addCleanup(environment.stop)
        self.model = SimpleNamespace(predict=Mock(return_value=[0.5]))
        self.factory = Mock(return_value=self.model)
        torch = SimpleNamespace(nn=SimpleNamespace(Sigmoid=lambda: "sigmoid"))
        modules = patch.dict(sys.modules, {
            "torch": torch, "sentence_transformers": SimpleNamespace(CrossEncoder=self.factory),
        })
        modules.start()
        self.addCleanup(modules.stop)

    def test_preloaded_model_is_reused_by_the_actual_factory_without_downloads(self):
        with patch("builtins.print"):
            preload_reranker()
        self.assertIs(get_reranker(), self.model)
        self.factory.assert_called_once()
        self.assertTrue(self.factory.call_args.kwargs["local_files_only"])
        self.assertEqual(self.factory.call_args.kwargs["max_length"], 512)
        self.model.predict.assert_called_once()

    def test_disabled_or_missing_model_does_not_prevent_api_startup(self):
        with patch.dict(os.environ, {"RAG_PRELOAD": "0"}):
            preload_reranker()
        self.factory.assert_not_called()
        self.factory.side_effect = OSError("not installed")
        with self.assertLogs("retrieval", level="WARNING"), patch("builtins.print"):
            preload_reranker()
        self.assertIsNone(get_reranker())
        self.factory.assert_called_once()

    def test_failed_warmup_does_not_crash_optional_search_startup(self):
        self.model.predict.side_effect = RuntimeError("unavailable kernel")
        with self.assertLogs("retrieval", level="ERROR"), patch("builtins.print"):
            preload_reranker()


class APIPreloadTests(unittest.TestCase):
    def test_http_and_polling_work_during_preparation_without_racing_inference(self):
        events = []
        started = threading.Event()
        release = threading.Event()

        def prepare():
            events.append('preparing')
            started.set()
            if not release.wait(5):
                raise RuntimeError('Test did not release preparation')
            events.append('ready')

        def answer(text, image):
            events.append(text)
            return {'answer': 'Ответ', 'intent': 'science'}

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(api.db, 'DB_FILE', str(Path(directory) / 'app.db')), \
             patch.object(api, 'preload_reranker', side_effect=prepare), \
             patch.object(api, 'run_agent', side_effect=answer) as model:
            with TestClient(api.app) as client:
                try:
                    self.assertTrue(started.wait(5))
                    self.assertEqual(client.get('/').status_code, 200)
                    self.assertEqual(client.get('/openapi.json').status_code, 200)
                    first = client.post('/ask/jobs', json={'text': 'first'}).json()
                    client.cookies.clear()  # A separate guest can wait in FIFO.
                    second = client.post('/ask/jobs', json={'text': 'second'}).json()
                    state = client.get('/ask/jobs/' + first['job_id']).json()
                    self.assertEqual(state['status'], 'queued')
                    self.assertTrue(state['preparing'])
                    model.assert_not_called()
                    release.set()
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        state = client.get('/ask/jobs/' + second['job_id']).json()
                        if state['status'] == 'done':
                            break
                        time.sleep(0.01)
                    self.assertEqual(state['status'], 'done')
                    self.assertFalse(state['preparing'])
                    self.assertEqual(events, ['preparing', 'ready', 'first', 'second'])
                finally:
                    release.set()


class QueuePreloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_optional_preparation_still_allows_processing(self):
        queue = InferenceQueue(lambda job: 'Ответ', initializer=Mock(side_effect=RuntimeError('not available')))
        with self.assertLogs('request_queue', level='ERROR'):
            await queue.start()
            try:
                job = queue.submit('Вопрос', None, None)
                await asyncio.wait_for(job.done.wait(), 5)
                self.assertEqual(queue.snapshot(job.id)['answer'], 'Ответ')
                self.assertFalse(queue.snapshot(job.id)['preparing'])
            finally:
                await queue.close()

    async def test_shutdown_during_preparation_does_not_run_waiting_questions(self):
        started = threading.Event()
        release = threading.Event()

        def prepare():
            started.set()
            release.wait(5)

        process = Mock(return_value='Ответ')
        queue = InferenceQueue(process, initializer=prepare)
        await queue.start()
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            job = queue.submit('Вопрос', b'photo', None)
            closing = asyncio.create_task(queue.close())
            await asyncio.sleep(0)
            self.assertEqual(job.status, 'failed')
            release.set()
            await asyncio.wait_for(closing, 5)
            process.assert_not_called()
            self.assertIsNone(job.image)
        finally:
            release.set()
            await queue.close()


if __name__ == "__main__":
    unittest.main()
