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


def _format_results(question, results, max_results):
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


def search_web(question, *, max_results=5):
    from itertools import islice
    query = search_query(question)
    max_results = max(1,min(int(max_results),10))
    backends = list(dict.fromkeys(x.strip() for x in os.getenv('WEB_SEARCH_BACKEND','bing,yandex').split(',') if x.strip()))[:3] or ['bing','yandex']
    if backends==['auto']:backends=['auto','bing','yandex']
    proxy = os.getenv('DDGS_PROXY') or os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY') or None
    verify = os.getenv('SSL_CERT_FILE') or os.getenv('REQUESTS_CA_BUNDLE') or True
    attempts=[]
    responded=False
    for backend in backends:
        try:
            client=DDGS(timeout=15, verify=verify, proxy=proxy)
            # Some clients fail while iterating, not in text() itself.
            results=list(islice(client.text(query,region='ru-ru',max_results=max_results,backend=backend),max_results*4))
            responded=True
            value=_format_results(question,results,max_results)
            attempts.append({'backend':backend,'status':'ok' if value['web_sources'] else 'empty'})
            if value['web_sources']:
                return {**value,'web_provider':backend,'web_attempts':attempts}
        except Exception as exc:
            empty='no results found' in str(exc).casefold()
            responded=responded or empty
            attempts.append({'backend':backend,'status':'empty' if empty else 'failed','error_type':type(exc).__name__})
            # Do not log query, proxy URLs, credential values or exception text.
            logger.warning('Web search backend=%s status=%s error_type=%s',backend,'empty' if empty else 'failed',type(exc).__name__)
    return {'web_context':'','web_sources':[],
            'web_error':None if responded else 'Поиск в интернете сейчас недоступен. Попробуйте повторить запрос позже.',
            'web_provider':None,'web_attempts':attempts}
