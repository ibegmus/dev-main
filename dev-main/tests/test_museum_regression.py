"""Regression tests against the repository's actual museum documents."""

import sqlite3
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "Video Project 9.txt"


def stored_documents():
    path = ROOT / "chroma_db_new" / "chroma.sqlite3"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        records = {}
        for identifier, key, value in connection.execute(
                "SELECT id, key, string_value FROM embedding_metadata WHERE string_value IS NOT NULL ORDER BY id, key"):
            record = records.setdefault(identifier, {"metadata": {}})
            if key == "chroma:document":
                record["text"] = value
            else:
                record["metadata"][key] = value
    return [Document(page_content=record["text"], metadata=record["metadata"])
            for record in records.values() if record.get("text")]


class SnapshotStore:
    def __init__(self, docs, semantic=None):
        self.docs = docs
        self.semantic = semantic or []

    def get(self):
        return {"documents": [doc.page_content for doc in self.docs],
                "metadatas": [doc.metadata for doc in self.docs]}

    def similarity_search(self, question, k):
        return self.semantic[:k]


def rejecting_reranker():
    return SimpleNamespace(predict=Mock(side_effect=lambda pairs, **kwargs: [0.01] * len(pairs)))


class MuseumEvidenceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = stored_documents()
        cls.store = SnapshotStore(cls.docs)
        cls.retriever = HybridRetriever(cls.store, reranker_factory=rejecting_reranker, min_score=0.9)

    def assert_georgia_evidence(self, result):
        context = result["local_context"]
        self.assertIn("1980", context)
        self.assertIn("Махарадзе", context)
        self.assertIn("чайной плантации", context)
        self.assertTrue(result["sources"])
        self.assertFalse(result["local_error"])

    def test_real_collection_keeps_georgia_despite_low_reranker_scores(self):
        self.assertGreater(len(self.docs), 700)
        for question in [
                "Когда в музее появилась почва из Грузии и чем она интересна?",
                "В каком году был взят монолит из Грузии?",
                "Расскажи про почву из Грузии в коллекции музея"]:
            with self.subTest(question=question), self.assertLogs("retrieval", level="WARNING"):
                result = self.retriever.search(question)
            self.assert_georgia_evidence(result)
            self.assertLess(result["local_relevance"], self.retriever.min_score)

    def test_bad_vector_ranking_cannot_crowd_out_exact_subject(self):
        unrelated = [doc for doc in self.docs if "Грузии" not in doc.page_content][:24]
        store = SnapshotStore(self.docs, semantic=unrelated)
        retriever = HybridRetriever(store, reranker_factory=rejecting_reranker)
        with self.assertLogs("retrieval", level="WARNING"):
            result = retriever.search("Когда в музее появилась почва из Грузии?")
        self.assert_georgia_evidence(result)

    def test_empty_vector_collection_uses_actual_local_book(self):
        retriever = HybridRetriever(SnapshotStore([]), source_path=BOOK,
                                    reranker_factory=lambda: None)
        self.assert_georgia_evidence(retriever.search("Почва из Грузии: когда появилась в музее?"))

    def test_partially_populated_collection_does_not_hide_book_facts(self):
        unrelated = next(doc for doc in self.docs if "Грузии" not in doc.page_content)
        retriever = HybridRetriever(SnapshotStore([unrelated]), source_path=BOOK,
                                    reranker_factory=lambda: None)
        self.assert_georgia_evidence(retriever.search("Когда в музее появилась почва из Грузии?"))

    def test_unavailable_vector_collection_uses_book_without_reindexing(self):
        store = SnapshotStore([])
        store.get = Mock(side_effect=ConnectionError("Collection unavailable"))
        store.similarity_search = Mock(side_effect=ConnectionError("Embedding model unavailable"))
        with self.assertLogs("retrieval", level="ERROR"):
            retriever = HybridRetriever(store, source_path=BOOK, reranker_factory=lambda: None)
            result = retriever.search("Когда привезли почву из Грузии?")
        self.assert_georgia_evidence(result)

    def test_other_real_local_subject_survives_same_filter(self):
        with self.assertLogs("retrieval", level="WARNING"):
            result = self.retriever.search("Какие почвы преобладают в Кытлымском почвенном районе?")
        self.assertIn("Кытлым", result["local_context"])
        self.assertIn("40 %", result["local_context"])
        self.assertIn("подзолист", result["local_context"])

    def test_missing_metadata_does_not_discard_document(self):
        store = SnapshotStore([])
        store.get = Mock(return_value={"documents": ["Монолит из Грузии привезли в 1980 году из Махарадзе, с чайной плантации."], "metadatas": []})
        retriever = HybridRetriever(store, reranker_factory=lambda: None)
        self.assert_georgia_evidence(retriever.search("Почва из Грузии"))


if __name__ == "__main__":
    unittest.main()
