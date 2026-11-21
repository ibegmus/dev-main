"""Hybrid retrieval with explicit, comparable reranker probabilities."""

import logging
import math
import os
import re
import time
from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from threading import Lock

from langchain_core.documents import Document
from nltk.stem.snowball import RussianStemmer
from rank_bm25 import BM25Plus

from model_runtime import configure_cpu_threads
from performance import timed_stage

logger = logging.getLogger(__name__)
_stemmer = RussianStemmer()
_stopwords = set("а без бы был была были было в вам вас ваш ваша ваши во вот вы где да для до его ее если есть еще же за зачем и из или им их к как какая какие какой когда кто ли мне мой мы на над надо нас наш не него нее нет но о об он она они оно от по под при про с со так такой там то у уже чем что чтобы это этот я".split())
_canonical_prefixes = {"музе": "музей", **{root: root for root in
                        ("чернозем", "краснозем", "желтозем", "серозем", "бурозем")}}
_generic_focus_prefixes = ("музей", "почв", "монолит", "образ", "коллекц", "появ", "интерес",
                           "поступ", "привез", "год", "расскаж", "покаж", "объясн")


def tokenize(text):
    words = re.findall(r"[\w]+", text.casefold().replace("ё", "е"))
    return [next((root for prefix, root in _canonical_prefixes.items() if word.startswith(prefix)), _stemmer.stem(word)) for word in words
            if word not in _stopwords and (len(word) > 2 or word.isdigit() or word == "ph")]


def doc_source(doc):
    metadata = doc.metadata
    headings = [metadata.get(key) for key in ("Header 1", "Header 2", "Header 3")]
    return str(metadata.get("title") or " / ".join(str(h) for h in headings if h)
               or metadata.get("source") or "Книга музея")


def load_source_documents(path):
    """Read the authoritative local book independently of the vector collection."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    source = Path(path)
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
    ).split_text(source.read_text(encoding="utf-8-sig"))
    docs = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(sections)
    for doc in docs:
        doc.metadata["source"] = source.name
    return docs


@lru_cache(maxsize=1)
def get_reranker():
    # A missing model/download must not disable lexical and vector retrieval.
    try:
        import torch
        from sentence_transformers import CrossEncoder

        configure_cpu_threads(torch)
        return CrossEncoder(os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
                            max_length=512, local_files_only=True)
    except (OSError, ImportError):
        logger.warning("Reranker is not cached; using hybrid retrieval without reranking")
        return None


def preload_reranker():
    """Prepare the existing local model before the API accepts questions."""
    if os.getenv("RAG_PRELOAD", "1").strip().lower() in {"0", "false", "off"}:
        return
    started = time.monotonic()
    print("Подготавливаем поиск: загружаем модель отбора фрагментов...", flush=True)
    try:
        model = get_reranker()
        if model is None:
            print("Поиск будет работать без дополнительной модели отбора.", flush=True)
            return
        import torch

        # Initialize inference kernels too, without reading user data or writing
        # an answer/history entry. This uses the same scoring path as search.
        model.predict(
            [("Проверка поиска", "Подготовка поиска по музейным материалам.")],
            activation_fn=torch.nn.Sigmoid(), batch_size=1, show_progress_bar=False,
        )
        print(f"Модель отбора готова за {time.monotonic() - started:.1f} с.", flush=True)
    except Exception:
        logger.exception("Search preparation failed; the API will continue with retrieval fallback")


class HybridRetriever:
    def __init__(self, vectorstore, *, reranker_factory=None, candidates=24,
                 top_k=6, min_score=0.1, max_chars=12000, source_path=None,
                 cache_ttl=None, cache_size=64):
        self.vectorstore = vectorstore
        self.reranker_factory = reranker_factory or get_reranker
        self.candidate_count = candidates
        self.top_k = top_k
        self.min_score = min_score
        self.max_chars = max_chars
        self.source_path = source_path
        self.cache_ttl = float(os.getenv("RAG_CACHE_TTL_SECONDS", "300") if cache_ttl is None else cache_ttl)
        if not math.isfinite(self.cache_ttl) or self.cache_ttl < 0:
            raise ValueError("RAG_CACHE_TTL_SECONDS must be a non-negative finite number")
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self._cache_lock = Lock()
        self._index = ([], None, [], True)
        self.rebuild()

    def rebuild(self):
        try:
            data = self.vectorstore.get()
        except Exception:
            logger.exception("Cannot read vector collection; checking local source document")
            data = {}
        texts = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        docs = [Document(page_content=text, metadata=(metadatas[i] or {}) if i < len(metadatas) else {})
                for i, text in enumerate(texts) if text and text.strip()]
        if self.source_path:
            try:
                book_docs = load_source_documents(self.source_path)
                stored_count = len(docs)
                seen = {" ".join(doc.page_content.split()) for doc in docs}
                for doc in book_docs:
                    key = " ".join(doc.page_content.split())
                    if key not in seen:
                        docs.append(doc)
                        seen.add(key)
                if not stored_count:
                    logger.warning("Vector collection is empty/unavailable; using %s chunks from local book", len(docs))
            except (OSError, UnicodeError):
                logger.exception("Local source document unavailable")
        corpus = [tokenize(doc_source(doc) + " " + doc.page_content) for doc in docs]
        bm25 = BM25Plus(corpus) if any(corpus) else None
        # Swap the entire index, keeping concurrent admin updates consistent.
        with self._cache_lock:
            self._index = (docs, bm25, [set(tokens) for tokens in corpus], not bool(docs))
            self._cache.clear()
        logger.info("Museum retrieval initialized with %s chunks", len(docs))

    def _cached_search(self, question, generation):
        if self.cache_ttl == 0 or self.cache_size <= 0:
            return None
        key = sha256(question.encode("utf-8")).digest()
        now = time.monotonic()
        with self._cache_lock:
            for expired in [k for k, (_, deadline, _) in self._cache.items() if deadline <= now]:
                del self._cache[expired]
            entry = self._cache.get(key)
            if entry is None or entry[0] is not generation:
                return None
            self._cache.move_to_end(key)
            return deepcopy(entry[2])

    def _remember_search(self, question, generation, result):
        if self.cache_ttl == 0 or self.cache_size <= 0:
            return
        # Keep the query digest and document excerpts, not the question or photo.
        key = sha256(question.encode("utf-8")).digest()
        with self._cache_lock:
            if generation is not self._index:
                return  # An admin refreshed the index while this search ran.
            self._cache[key] = (generation, time.monotonic() + self.cache_ttl, deepcopy(result))
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def candidates(self, question):
        docs, failed, _ = self._candidates(question)
        return docs, failed

    def _candidates(self, question):
        docs, bm25, terms, unavailable = self._index
        vector_failed = False
        try:
            with timed_stage("vector_search"):
                semantic = self.vectorstore.similarity_search(question, k=self.candidate_count)
        except Exception:
            logger.exception("Vector retrieval unavailable; using lexical retrieval")
            semantic = []
            vector_failed = True
        lexical = []
        anchors = []
        if bm25:
            query_terms = set(tokenize(question))
            scores = bm25.get_scores(list(query_terms))
            ordered = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            lexical = [docs[i] for i in ordered if query_terms & terms[i]][:self.candidate_count]
            # Rare subject terms (a place, person, soil type) are stronger evidence
            # than generic overlap on "museum" or "soil". Preserve these matches
            # even when the semantic model or reranker ranks them poorly.
            focus = {term for term in query_terms
                     if not term.startswith(_generic_focus_prefixes)}
            rare_cutoff = math.log((len(docs) + 1) / max(1, len(docs) * 0.1))
            rare = {term for term in focus if bm25.idf.get(term, 0) >= rare_cutoff}
            anchored_texts = set()
            for i in ordered:
                key = " ".join(docs[i].page_content.split())
                if rare & terms[i] and key not in anchored_texts:
                    anchors.append(docs[i])
                    anchored_texts.add(key)
                if len(anchors) >= min(3, self.top_k, self.candidate_count):
                    break

        fused = {}
        for ranking in (semantic, lexical):
            seen = set()
            for rank, doc in enumerate(ranking):
                key = " ".join(doc.page_content.split())
                if key in seen:
                    continue
                seen.add(key)
                previous = fused.get(key, (0, doc))[0]
                fused[key] = (previous + 1 / (60 + rank), doc)
        ranked = sorted(fused.values(), key=lambda item: item[0], reverse=True)
        pool, seen = [], set()
        for doc in anchors + [doc for _, doc in ranked]:
            key = " ".join(doc.page_content.split())
            if key not in seen:
                pool.append(doc)
                seen.add(key)
        return pool[:self.candidate_count], vector_failed or unavailable, anchors

    def search(self, question):
        generation = self._index
        cached = self._cached_search(question, generation)
        if cached is not None:
            with timed_stage("local_cache_hit"):
                return cached
        candidates, vector_failed, anchors = self._candidates(question)
        if not candidates:
            return {"local_context": "", "sources": [], "local_relevance": 0.0,
                    "local_error": vector_failed}
        cacheable = False
        try:
            import torch

            pairs = [(question, doc_source(doc) + "\n" + doc.page_content) for doc in candidates]
            with timed_stage("reranker_init") as details:
                model = self.reranker_factory()
                if details is not None:
                    details["cpu_threads"] = torch.get_num_threads()
            scores = None
            if model is not None:
                with timed_stage("rerank"):
                    scores = model.predict(
                        pairs, activation_fn=torch.nn.Sigmoid(), batch_size=4, show_progress_bar=False,
                    )
            if scores is None:
                chosen = [(doc, None) for doc in candidates]
                relevance = None
            else:
                if len(scores) != len(candidates):
                    raise ValueError("Reranker returned an unexpected number of scores")
                ranked = sorted(zip(candidates, map(float, scores)), key=lambda item: item[1], reverse=True)
                chosen = [(doc, score) for doc, score in ranked if score >= self.min_score]
                relevance = ranked[0][1]
                # A score cutoff is not evidence that a documented fact is absent.
                # Keep exact subject matches ahead of less specific semantic hits.
                score_by_text = {doc.page_content: score for doc, score in ranked}
                anchor_texts = {doc.page_content for doc in anchors}
                chosen = [(doc, score_by_text[doc.page_content]) for doc in anchors] + [
                    (doc, score) for doc, score in chosen if doc.page_content not in anchor_texts
                ]
                if anchors and all(score_by_text[doc.page_content] < self.min_score for doc in anchors):
                    logger.warning("Reranker rejected lexical subject matches; retained %s local sources", len(anchors))
                cacheable = not vector_failed
        except Exception:
            logger.exception("Reranker unavailable; preserving hybrid search candidates")
            chosen = [(doc, None) for doc in candidates]
            relevance = None

        sections, sources, used = [], [], 0
        for doc, score in chosen[:self.top_k]:
            source = doc_source(doc)
            label = f"M{len(sections) + 1}"
            section = f"[{label}] {source}\n{doc.page_content}"
            if used + len(section) > self.max_chars:
                break
            sections.append(section)
            sources.append(f"[{label}] {source}")
            used += len(section) + 2
            logger.info("Local source %s, relevance=%s", label, score)
        result = {"local_context": "\n\n".join(sections), "sources": sources,
                  "local_relevance": relevance, "local_error": False}
        if cacheable:
            self._remember_search(question, generation, result)
        return result
