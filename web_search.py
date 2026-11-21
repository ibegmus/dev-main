"""Public web search with URLs, bounded snippets and visible failure state."""

import html
import logging
import os
import re
from urllib.parse import urlsplit, urlunsplit

from ddgs import DDGS

logger = logging.getLogger(__name__)


def wants_web(question):
    return bool(re.search(
        r"интернет|онлайн|\bв сети\b|загугл|поищи|"
        r"(?:свеж|новейш|последн)\w*\s+(?:исследован|стать|данн)|"
        r"на сегодняшний день|актуальн\w*\s+данн",
        question, re.IGNORECASE,
    ))


def museum_history(question):
    return bool(re.search(r"музе|монолит|экспонат|коллекц", question, re.IGNORECASE)
                and re.search(r"когда|откуда|кто|в каком году|истори|появил|привез|привёз", question, re.IGNORECASE))


def search_query(question):
    query = re.sub(r"(?:найди|найдите|поищи|поищите|посмотри|проверь)\s+(?:в\s+)?(?:интернете|сети)",
                   "", question, flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query).strip(" ,.:;?!")
    return query or question.strip()


def _text(value, limit):
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()[:limit]


def search_web(question, *, max_results=5):
    query = search_query(question)
    try:
        client = DDGS(timeout=15, verify=os.getenv("SSL_CERT_FILE") or True)
        results = client.text(query, region="ru-ru", max_results=max_results,
                              backend=os.getenv("WEB_SEARCH_BACKEND", "auto"))
    except Exception as exc:
        logger.warning("Web search failed: %s", type(exc).__name__)
        no_results = "no results found" in str(exc).casefold()
        return {"web_context": "", "web_sources": [],
                "web_error": None if no_results else "Поиск в интернете сейчас недоступен. Попробуйте повторить запрос позже."}

    sections, sources, seen = [], [], set()
    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("href") or item.get("url") or "").strip()
        try:
            parts = urlsplit(url)
            if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
                continue
            url = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
        except ValueError:
            continue
        snippet = _text(item.get("body") or item.get("snippet"), 1200)
        if not snippet or url in seen:
            continue
        label = f"W{len(sections) + 1}"
        title = _text(item.get("title"), 240) or parts.hostname
        if museum_history(question) and not re.search(
                r"почв|монолит|soil|pedolog|ургaу|ургау|urgau", title + " " + snippet, re.IGNORECASE):
            # A museum with the same country/name is not evidence about our soil museum.
            continue
        seen.add(url)
        sections.append(f"[{label}] {title}\nURL: {url}\nФрагмент: {snippet}")
        sources.append({"id": label, "title": title, "url": url})
        if len(sections) >= max_results:
            break
    return {"web_context": "\n\n".join(sections), "web_sources": sources, "web_error": None}
