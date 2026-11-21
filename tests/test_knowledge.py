import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage

from performance import request_profile
from retrieval import HybridRetriever, doc_source, tokenize
from web_search import search_query, search_web, wants_web


GEORGIA = Document(
    page_content="Самый крайний монолит взят в Грузии, на чайной плантации, в 1980 году. Тогда был Всесоюзный съезд в Тбилиси. Потом всех повезли в город Махарадзе — это Аджария. Из Грузии везли землю, а не фрукты и вино.",
    metadata={"Header 1": "Почвенный музей", "Header 2": "Истории о сборе монолитов"},
)
KYTLYM = Document(
    page_content="Доминируют горные подзолистые почвы (40 %) с большим участием горных лесных бурых (до 20 %). Заметно участие горных луговых почв (до 20 %).",
    metadata={"Header 2": "Кытлымский почвенный район"},
)
IRRELEVANT = Document(page_content="Растительность пустынь и песчаные дюны.", metadata={"title": "Пустыни"})


class FakeStore:
    def __init__(self, docs, semantic=None):
        self.docs = docs
        self.semantic = semantic if semantic is not None else docs

    def get(self):
        return {"documents": [d.page_content for d in self.docs],
                "metadatas": [d.metadata for d in self.docs]}

    def similarity_search(self, question, k):
        return self.semantic[:k]


class RetrievalTests(unittest.TestCase):
    def test_normalizes_russian_word_forms_and_yo(self):
        self.assertEqual(tokenize("чернозёмами"), tokenize("чернозем"))
        self.assertEqual(tokenize("музея"), tokenize("музей"))
        self.assertNotIn("как", tokenize("Как образуются почвы?"))

    def test_headings_are_part_of_lexical_search(self):
        store = FakeStore([GEORGIA, KYTLYM, IRRELEVANT], semantic=[])
        r = HybridRetriever(store, reranker_factory=lambda: None)
        candidates, _ = r.candidates("Какие почвы Кытлымского района?")
        self.assertIn(KYTLYM.page_content, [d.page_content for d in candidates])

    def test_reranker_probabilities_drop_unrelated_passages(self):
        model = SimpleNamespace(predict=Mock(return_value=[0.97, 0.02]))
        r = HybridRetriever(FakeStore([GEORGIA, IRRELEVANT]), reranker_factory=lambda: model)
        result = r.search("Когда привезли монолит из Грузии?")
        self.assertIn("1980", result["local_context"])
        self.assertNotIn(IRRELEVANT.page_content, result["local_context"])
        self.assertEqual(result["local_relevance"], 0.97)
        self.assertEqual(type(model.predict.call_args.kwargs["activation_fn"]).__name__, "Sigmoid")

    def test_embedding_failure_preserves_lexical_results(self):
        store = FakeStore([GEORGIA, KYTLYM, IRRELEVANT])
        store.similarity_search = Mock(side_effect=ConnectionError("Ollama offline"))
        r = HybridRetriever(store, reranker_factory=lambda: None)
        with self.assertLogs("retrieval", level="ERROR"):
            result = r.search("В каком году монолит взят в Грузии?")
        self.assertIn("1980", result["local_context"])
        self.assertFalse(result["local_error"])

    def test_missing_reranker_keeps_context(self):
        r = HybridRetriever(FakeStore([GEORGIA]), reranker_factory=lambda: None)
        result = r.search("Грузия")
        self.assertIn("Махарадзе", result["local_context"])
        self.assertIsNone(result["local_relevance"])

    def test_empty_or_unrelated_collection_does_not_invent_context(self):
        r = HybridRetriever(FakeStore([]), reranker_factory=lambda: None)
        self.assertEqual(r.search("Грузия")["local_context"], "")
        model = SimpleNamespace(predict=Mock(return_value=[0.001]))
        r = HybridRetriever(FakeStore([IRRELEVANT]), reranker_factory=lambda: model)
        self.assertEqual(r.search("Грузия")["local_context"], "")

    def test_budget_and_duplicate_passages(self):
        duplicate = Document(page_content=GEORGIA.page_content, metadata=GEORGIA.metadata)
        model = SimpleNamespace(predict=Mock(return_value=[0.9, 0.8]))
        r = HybridRetriever(FakeStore([GEORGIA, duplicate, IRRELEVANT]),
                            reranker_factory=lambda: model, max_chars=300)
        result = r.search("Грузия")
        self.assertEqual(len(result["sources"]), 1)
        self.assertLessEqual(len(result["local_context"]), 300)

    def test_admin_refresh_rebuilds_index(self):
        store = FakeStore([IRRELEVANT], semantic=[])
        r = HybridRetriever(store, reranker_factory=lambda: None)
        self.assertEqual(r.candidates("Грузия")[0], [])
        store.docs.append(GEORGIA)
        r.rebuild()
        self.assertIn(GEORGIA.page_content, [d.page_content for d in r.candidates("Грузия")[0]])


class WebSearchTests(unittest.TestCase):
    @patch("web_search.DDGS")
    def test_links_snippets_and_deduplication(self, client):
        client.return_value.text.return_value = [
            {"title": "Soil &amp; carbon", "href": "https://example.org/soil_carbon#part", "body": "Organic matter forms from residues."},
            {"title": "Duplicate", "href": "https://example.org/soil_carbon", "body": "Duplicate"},
            {"title": "Unsafe", "href": "javascript:alert(1)", "body": "Unsafe"},
            {"title": "Empty", "href": "https://example.org/empty", "body": ""},
        ]
        result = search_web("Найди в интернете, как образуется гумус")
        self.assertEqual(len(result["web_sources"]), 1)
        self.assertEqual(result["web_sources"][0]["url"], "https://example.org/soil_carbon")
        self.assertIn("Organic matter", result["web_context"])
        self.assertIn("Soil & carbon", result["web_context"])
        self.assertIsNone(result["web_error"])
        self.assertEqual(client.return_value.text.call_args.args[0], "как образуется гумус")
        self.assertGreater(client.call_args.kwargs["timeout"], 8)
        self.assertIsNot(client.call_args.kwargs["verify"], False)

    @patch("web_search.DDGS")
    def test_outage_is_distinct_from_no_results(self, client):
        client.return_value.text.side_effect = TimeoutError("Search timed out")
        with self.assertLogs("web_search", level="WARNING"):
            failed = search_web("почвы")
        self.assertIn("недоступен", failed["web_error"])
        client.return_value.text.side_effect = None
        client.return_value.text.return_value = []
        empty = search_web("почвы")
        self.assertIsNone(empty["web_error"])
        self.assertEqual(empty["web_sources"], [])

    def test_explicit_and_fresh_queries(self):
        self.assertTrue(wants_web("Найди в интернете статьи о чернозёме"))
        self.assertTrue(wants_web("Последние исследования почвенного углерода"))
        self.assertFalse(wants_web("Когда в музее появился монолит из Грузии?"))
        self.assertEqual(search_query("найдите в сети, гумус"), "гумус")

    @patch("web_search.DDGS")
    def test_namesake_museum_is_not_used_as_soil_museum_evidence(self, client):
        client.return_value.text.return_value = [
            {"title": "Государственный музей Грузии", "href": "https://example.org/other", "body": "Музей был основан в 1981 году в Москве."},
            {"title": "Почвенный музей", "href": "https://example.org/soil", "body": "Почва из Грузии, монолит 1980 года."},
        ]
        result = search_web("Когда в музее появилась почва из Грузии?")
        self.assertEqual([s["url"] for s in result["web_sources"]], ["https://example.org/soil"])
        self.assertNotIn("1981", result["web_context"])


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the actual graph without opening/migrating tracked Chroma files.
        path = Path(__file__).resolve().parents[1] / "agent.py"
        spec = importlib.util.spec_from_file_location("knowledge_agent_under_test", path)
        cls.agent = importlib.util.module_from_spec(spec)
        fake_rag = SimpleNamespace(DB_PATH="unused", DOC_PATH=None, embeddings=None)
        with patch.dict(sys.modules, {"rag": fake_rag}), patch("langchain_chroma.Chroma", return_value=FakeStore([])):
            spec.loader.exec_module(cls.agent)

    def setUp(self):
        self.a = self.agent
        for attribute in ["llm", "classifier_llm"]:
            replacement = patch.object(self.a, attribute, Mock())
            replacement.start()
            self.addCleanup(replacement.stop)

    def model_json(self, data):
        return AIMessage(content=json.dumps(data, ensure_ascii=False))

    def test_greeting_before_question_still_uses_retrieval(self):
        with patch.object(self.a.classifier_llm, "invoke", return_value=self.model_json({"intent": "science"})):
            self.assertEqual(self.a.classify_intent({"question": "Здравствуйте! Когда привезли почву из Грузии?"})["intent"], "science")
        self.assertEqual(self.a.classify_intent({"question": "Здравствуйте!"})["intent"], "small_talk")

    def test_classifier_prose_does_not_trigger_substring_refusal(self):
        with patch.object(self.a.classifier_llm, "invoke", return_value=AIMessage(content="Не ОТКАЗ, вопрос о почвах")):
            with self.assertLogs("knowledge_agent_under_test", level="WARNING"):
                result = self.a.classify_intent({"question": "Почвы"})
        self.assertEqual(result["intent"], "science")

    def test_explicit_web_search_runs_even_with_local_context_and_photo(self):
        state = {"question": "Найди в интернете свойства почвы", "local_context": "Текст музея", "image_description": "Коричневая почва"}
        self.assertEqual(self.a.route_web(state), "academic")

    def test_catalog_lookup_does_not_bypass_explicit_web_request(self):
        self.assertEqual(self.a.route_after_tools({
            "question": "Сравни монолиты в каталоге и найди в интернете свойства почв",
            "catalog_context": "Монолиты найдены",
        }), "local")

    def test_partial_context_requests_web(self):
        with patch.object(self.a.classifier_llm, "invoke", return_value=self.model_json({"sufficient": False})):
            result = self.a.assess_context({"question": "Год и новые исследования?", "local_context": "Монолит из Грузии"})
        self.assertTrue(result["needs_web"])
        self.assertEqual(self.a.route_after_assessment(result), "academic")

    def test_complete_context_avoids_unnecessary_web_search(self):
        with patch.object(self.a.classifier_llm, "invoke", return_value=self.model_json({"sufficient": True})):
            result = self.a.assess_context({"question": "Когда?", "local_context": "В 1980 году"})
        self.assertFalse(result["needs_web"])

    def test_museum_history_uses_local_evidence_unless_web_is_requested(self):
        state = {"question": "Когда в музее появилась почва из Грузии?", "local_context": "[M1] Монолит 1980 года"}
        self.assertEqual(self.a.route_web(state), "generate")
        state["question"] += " Найди в интернете подтверждение."
        self.assertEqual(self.a.route_web(state), "academic")

    def test_missing_local_history_does_not_automatically_search_other_museums(self):
        state = {"question": "Когда в музее появилась почва из Грузии?", "local_context": ""}
        self.assertEqual(self.a.route_web(state), "generate")
        answer = self.a.generate_answer(state)["answer"]
        self.assertIn("Это не означает", answer)
        self.assertNotIn("Поиск в интернете тоже", answer)

    def test_web_pages_alone_cannot_assert_our_collection_is_missing_a_sample(self):
        state = {"question": "Когда появилась почва из Грузии в нашем музее?",
                 "local_context": "", "web_context": "[W1] Другой музей был основан в 1904 году.",
                 "web_sources": [{"id": "W1", "title": "Другой музей", "url": "https://example.org"}]}
        result = self.a.generate_answer(state)["answer"]
        self.assertIn("Это не означает", result)
        self.assertNotIn("1904", result)
        self.a.llm.invoke.assert_not_called()

    def test_actual_georgia_passage_reaches_answer_model_and_skips_web(self):
        from test_museum_regression import SnapshotStore, rejecting_reranker, stored_documents

        retriever = HybridRetriever(SnapshotStore(stored_documents()), reranker_factory=rejecting_reranker)
        with patch.object(self.a, "retriever", retriever), \
             patch.object(self.a.classifier_llm, "invoke", return_value=self.model_json({"intent": "science"})), \
             patch.object(self.a.llm, "invoke", return_value=AIMessage(content="Ответ по предоставленному источнику")) as answer, \
             patch.object(self.a, "search_web") as web, \
             self.assertLogs("retrieval", level="WARNING"):
            result = self.a.app.invoke({"question": "Когда в музее появилась почва из Грузии и чем она интересна?"})
        materials = answer.call_args.args[0][1].content
        for fact in ["1980", "Махарадзе", "чайной плантации"]:
            self.assertIn(fact, materials)
        web.assert_not_called()
        self.assertTrue(result["sources"])
        self.assertFalse(result.get("web_sources"))

    def test_timings_follow_actual_graph_and_retrieval_execution(self):
        reranker = SimpleNamespace(predict=Mock(return_value=[0.9]))
        retriever = HybridRetriever(FakeStore([GEORGIA]), reranker_factory=lambda: reranker)
        with patch.object(self.a, "retriever", retriever), \
             patch.object(self.a.classifier_llm, "invoke", return_value=self.model_json({"intent": "science"})), \
             patch.object(self.a.llm, "invoke", return_value=AIMessage(content="В 1980 году")), \
             self.assertLogs("terramind.performance", level="INFO") as logs:
            with request_profile("graph-id", submitted_at=time.monotonic()):
                result = self.a.app.invoke({"question": "Когда в музее появилась почва из Грузии?"})
        text = "\n".join(logs.output)
        for stage in ["classify", "vector_search", "reranker_init", "rerank", "local_search", "answer"]:
            self.assertIn(f"{stage}=", text)
        self.assertIn("1980", result["answer"])
        self.assertNotIn("Грузии", text)

    def test_repeated_local_search_still_runs_web_and_generates_a_new_answer(self):
        reranker = SimpleNamespace(predict=Mock(return_value=[0.9]))
        store = FakeStore([GEORGIA])
        store.similarity_search = Mock(wraps=store.similarity_search)
        retriever = HybridRetriever(store, reranker_factory=lambda: reranker)
        web_results = [
            {"web_context": f"[W1] Свежие данные {number}", "web_error": None,
             "web_sources": [{"id": "W1", "title": f"Статья {number}", "url": f"https://example.org/{number}"}]}
            for number in (1, 2)
        ]
        with patch.object(self.a, "retriever", retriever), \
             patch.object(self.a.classifier_llm, "invoke", return_value=self.model_json({"intent": "science"})), \
             patch.object(self.a, "search_web", side_effect=web_results) as web, \
             patch.object(self.a.llm, "invoke", side_effect=[AIMessage(content="Первый ответ"), AIMessage(content="Второй ответ")]) as answer, \
             self.assertLogs("terramind.performance", level="INFO") as logs:
            with request_profile("repeat-id", submitted_at=time.monotonic()):
                first = self.a.app.invoke({"question": "Найди в интернете дополнение о почве из Грузии"})
                second = self.a.app.invoke({"question": "Найди в интернете дополнение о почве из Грузии"})
        self.assertIn("local_cache_hit=", "\n".join(logs.output))
        store.similarity_search.assert_called_once()
        reranker.predict.assert_called_once()
        self.assertEqual(web.call_count, 2)
        self.assertEqual(answer.call_count, 2)
        self.assertIn("Свежие данные 2", answer.call_args.args[0][1].content)
        self.assertIn("1980", answer.call_args.args[0][1].content)
        self.assertIn("https://example.org/1", first["answer"])
        self.assertIn("https://example.org/2", second["answer"])

    def test_generation_combines_all_evidence_and_lists_real_links(self):
        state = {"question": "Объясните", "local_context": "[M1] Грузия, 1980 год",
                 "sources": ["[M1] История музея"], "catalog_context": "Гумус: 8%",
                 "web_context": "[W1] Organic carbon", "image_description": "Тёмный верхний слой",
                 "web_sources": [{"id": "W1", "title": "Carbon", "url": "https://example.org/soil_carbon"}]}
        with patch.object(self.a.llm, "invoke", return_value=AIMessage(content="Объяснение")) as invoke:
            answer = self.a.generate_answer(state)["answer"]
        messages = invoke.call_args.args[0]
        self.assertIsInstance(messages[0], SystemMessage)
        for fact in ["1980", "Гумус: 8%", "Organic carbon", "Тёмный верхний слой"]:
            self.assertIn(fact, messages[1].content)
        self.assertIn("https://example.org/soil_carbon", answer)
        self.assertIn("[M1]", answer)
        self.assertIn("[K1]", answer)

    def test_search_outage_does_not_claim_internet_has_no_answer(self):
        result = self.a.generate_answer({"question": "Почвы", "web_error": "Поиск в интернете сейчас недоступен."})["answer"]
        self.assertIn("недоступен", result)
        self.assertNotIn("тоже не дал", result)

    def test_graph_runs_web_and_preserves_museum_context(self):
        local = {"local_context": "[M1] Монолит привезён в 1980 году", "sources": ["[M1] История"],
                 "local_relevance": 0.9, "local_error": False}
        web = {"web_context": "[W1] Новые данные", "web_sources": [{"id": "W1", "title": "Статья", "url": "https://example.org/paper"}], "web_error": None}
        with patch.object(self.a.classifier_llm, "invoke", return_value=self.model_json({"intent": "science"})), \
             patch.object(self.a.retriever, "search", return_value=local), \
             patch.object(self.a, "search_web", return_value=web) as search, \
             patch.object(self.a.llm, "invoke", return_value=AIMessage(content="Ответ")) as invoke:
            result = self.a.app.invoke({"question": "Найди в интернете дополнение об этом монолите"})
        search.assert_called_once()
        self.assertIn("1980", invoke.call_args.args[0][1].content)
        self.assertIn("Новые данные", invoke.call_args.args[0][1].content)
        self.assertIn("https://example.org/paper", result["answer"])

    def test_empty_catalog_continues_to_museum_and_missing_facts_to_web(self):
        responses = [self.model_json({"intent": "tools"}), self.model_json({"tool": "find", "args": {"query": "Грузия"}})]
        with patch.object(self.a.classifier_llm, "invoke", side_effect=responses), \
             patch.object(self.a, "run_tool", return_value=""), \
             patch.object(self.a.retriever, "search", return_value={"local_context": "", "sources": []}) as local, \
             patch.object(self.a, "search_web", return_value={"web_context": "", "web_sources": [], "web_error": None}) as web:
            result = self.a.app.invoke({"question": "Найди монолит из Грузии"})
        local.assert_called_once()
        web.assert_called_once()
        self.assertIn("не нашёл", result["answer"])


if __name__ == "__main__":
    unittest.main()
