"""Durable, user-only observations for an advisory soil identification prompt.

This module stores reports, never diagnoses. ``fields`` may contain the existing
interview's extracted rows; a new row is accepted only if its quote occurs in the
latest visitor message. Photos, generated descriptions and assistant messages are
deliberately not inputs to the evidence recorder.
"""

import re

from soil_interview import LABELS, QUESTIONS, UNKNOWN, direct_answer, update_fields
from soil_observations import quote_is_statement, sentences


EXTRA_LABELS = {
    "surface_color": "Цвет, описанный посетителем",
    "texture_description": "Описание состава без ручной проверки",
    "organic_description": "Гипотеза посетителя об органическом веществе по внешнему виду (не анализ)",
}
FIELD_LABELS = {
    **LABELS, **EXTRA_LABELS,
    "water": "Обычный водный режим",
    "texture": "Механический состав по ручной проверке",
    "surface_water": "Наличие воды на снимке",
    "organic_layer": "Органический материал (сообщение без подтверждения анализом)",
}
ACCESS_LABELS = {
    "photo_only": "только фото / нет доступа к участку",
    "on_site": "есть доступ к участку / образцу",
    "unknown": "доступ неизвестен",
}
_UNKNOWN = re.compile(
    UNKNOWN.pattern
    + r"|не (?:пробовал[аи]?|трогал[аи]?|щупал[аи]?|проверяла|измеряла|мочил[аи]?|увлажнял[аи]?)"
    + r"|не уверен|неизвест|не могу (?:попасть|добраться|проверить)", re.I
)
_BARE_UNKNOWN = re.compile(
    r"(?:я )?(?:не знаю|не помню|не проверял[аи]?|не пробовал[аи]?|"
    r"не могу проверить|неизвестно)", re.I
)
_BRIEF_ORIGIN = re.compile(r"местн(?:ая|ый|ое)|(?:привозн|завозн)(?:ая|ой|ое)", re.I)
_HYPOTHETICAL = re.compile(
    r"\bесли\b|\bкогда\b|\bбуду\b|наверн|возможно|кажется|предполож", re.I
)
_ATTRIBUTED = re.compile(
    r"(?:вы|ты|ассистент|модель|бот)\s+(?:сказал|написал|ответил|счита)"
    r"|не\s*правда|не\s*верно|не говорил|не писала|описание (?:фото|модели)", re.I
)
_PHOTO = re.compile(r"\bфото\w*|\bсним\w*|на вид|выгляд|похож", re.I)
_TACTILE = re.compile(
    r"на ощупь|в руках|руками|пощуп|трогал|щупал|ощущ|увлажн|намоч|мочил|скат|шарик|лент"
    r"|механическ.{0,20}анализ|анализ.{0,20}механическ", re.I
)
_TEXTURE_WORDS = re.compile(r"пес|глин|суглин|супес|лип|рассып|шарик|лент|крош", re.I)
_STRUCTURE = re.compile(r"комоч|комков|глыб|столб|зернист|трещ", re.I)
_COLOR = re.compile(
    r"(?:т[её]мн|ч[её]рн).{0,25}(?:земл|сло|почв|поверхност)"
    r"|(?:земл|сло|почв|поверхност).{0,25}(?:т[её]мн|ч[её]рн)", re.I
)
_ORGANIC = re.compile(
    r"органическ|торф|гумус|растительн\w*\s+остат|минеральн|волокн|компост|перегной", re.I
)
_INFERENCE = re.compile(r"\bзначит\b|\bследовательно\b|\bпоэтому\b|означа\w*|говорит о", re.I)
_REGIME = re.compile(
    r"(?:обычно|всегда|постоянно).{0,35}(?:вод|сыр|мокр|сух)"
    r"|(?:вод|сыр|мокр|сух).{0,35}(?:обычно|всегда|постоянно)"
    r"|(?:быстро|долго|медленно).{0,15}(?:уход|стоит|впит|сохн|просых)"
    r"|(?:стоит|застаивается).{0,20}(?:недел|месяц|долго)"
    r"|после дожд.{0,40}(?:уход|стоит|впит|сохн|просых)"
    r"|не застаивается|не уходит|не просыхает", re.I
)
_WATER_WORDS = re.compile(r"вод|увлажн|сыр|мокр|сух|просых|дожд", re.I)
_VISIBLE_WATER = re.compile(r"вод|луж", re.I)
_REGION = re.compile(
    r"(?:[А-ЯЁ][а-яё-]+\s+){1,3}(?:область|край)"
    r"|[Рр]еспублика\s+[А-ЯЁ][а-яё-]+"
)
_REGION_UNCERTAINTY = re.compile(r"(?:не знаю|не уверен[а]?|неизвестно)\s*[,:\-–]?\s*$", re.I)
_LOCATION_UNKNOWN = re.compile(
    r"\b(?:место|местонахождение)\s+(?:(?:точно|точнее)\s+)?(?:неизвестно|не знаю)"
    r"|\bне знаю\s*,?\s*где\s+(?:сделан[оа]?|снят[оа]?)\s+(?:это\s+)?(?:фото\w*|сним\w*)",
    re.I,
)
_LOCATION_RETRACTION = re.compile(
    r"\bошиб(?:ся|лась)\b.{0,40}\b(?:регион|место|местонахождение)\b.{0,25}(?:неизвест|не знаю)"
    r"|\b(?:участок|место|фото)\s+не\s+в\s+[^.!?;]{1,60}\b(?:област[ьи]|кра[ея]|республик[аеуи])\b"
    r".{0,30}\b(?:место|регион)\b.{0,25}(?:неизвест|не знаю)",
    re.I,
)


def _clean(value):
    return " ".join(value.split())


def _compact_location_answer(text):
    """Read a short place answer, without geocoding it to a country or region."""
    short = _clean(text).strip(" .!")
    if len(short) > 80 or any(pattern.search(short) for pattern in (
        _UNKNOWN, _HYPOTHETICAL, _ATTRIBUTED,
    )):
        return None
    match = re.fullmatch(r"(?:(в|во|из|город|г\.|рядом с)\s+)?([А-ЯЁа-яё][А-ЯЁа-яё -]*)", short, re.I)
    if not match:
        return None
    prefix, place = match.groups()
    if place.casefold() in {"да", "нет", "есть", "могу", "привет", "спасибо", "продолжим", "дальше"}:
        return None
    normalized = "Екатеринбург" if place.casefold() == "екб" else None
    # Unknown abbreviations and ordinary lowercase prose are not guessed places.
    proper_name = re.fullmatch(r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁа-яё][а-яё]+)*(?: [А-ЯЁ][а-яё]+(?:-[А-ЯЁа-яё][а-яё]+)*){0,3}", place)
    if not normalized and not proper_name:
        return None
    return {"explicit": bool(prefix), "normalized": normalized}


def _organic_inference(quote):
    """An organic conclusion drawn from appearance is not a material report."""
    for match in _INFERENCE.finditer(quote):
        premise, conclusion = quote[:match.start()], quote[match.end():]
        if (_PHOTO.search(premise) or _COLOR.search(premise)) and _ORGANIC.search(conclusion):
            return True
    return False


def _exact_quote(message, quote):
    """Resolve the existing extractor's whitespace-normalized quote to raw text."""
    if not isinstance(quote, str) or not quote.strip():
        return None
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    match = re.search(pattern, message, re.I)
    return match.group() if match else None


def _rows(value):
    if not isinstance(value, dict):
        return {}
    return {
        key: {"status": row["status"], "quote": row["quote"], **(
            {"question": row["question"]}
            if (key in QUESTIONS and row.get("question") == QUESTIONS[key])
            or (key == "location_detail" and row.get("question") == QUESTIONS["location"]) else {}
        ), **(
            {"normalized": "Екатеринбург"}
            if key in {"location", "location_detail"} and row["status"] == "known"
            and (_compact_location_answer(row["quote"]) or {}).get("normalized") else {}
        )}
        for key, row in value.items()
        if key in FIELD_LABELS and isinstance(row, dict)
        and isinstance(row.get("status"), str) and row["status"] in {"known", "unknown"}
        and isinstance(row.get("quote"), str) and 1 <= len(row["quote"]) <= 1000
    }


def _evidence(value):
    value = value if isinstance(value, dict) else {}
    nested = value.get("evidence")
    if isinstance(nested, dict):
        value = nested
    # Legacy case fields may contain old model assignments. They must pass the
    # latest-message quote checks through fields= rather than seed durable facts.
    return value if value.get("kind") == "user_soil_evidence" and value.get("version") == 1 else {}


def _access_update(message):
    choices = []
    patterns = (
        ("photo_only", r"(?<!не )\b(?:только|лишь)\s+(?:(?:эта|это|этот|эти|одно|один|одна|несколько|\d+)\s+)?(?:фото\w*|сним\w*|изображени\w*)"),
        ("photo_only", r"не могу\s+(?:попасть|добраться|доехать|приехать|выехать).{0,35}(?:участ|мест|туда|дач)"),
        ("photo_only", r"(?:участ|мест|дач).{0,30}(?:попасть|добраться|доехать)\s+не могу"),
        ("photo_only", r"нет доступа.{0,25}(?:участ|мест|дач)|(?:участ|мест).{0,25}(?:недоступ|доступа нет)"),
        ("photo_only", r"доступа\s+к\s+(?:участку|месту|даче)\s+нет"),
        ("on_site", r"\b(?:теперь|сейчас|уже|сегодня)\s+(?:я\s+)?(?:нахожусь\s+)?на\s+(?:участке|месте|даче)\b"),
        ("on_site", r"\bя\s+(?:(?:сейчас|теперь|уже)\s+)?(?:нахожусь\s+)?на\s+(?:участке|месте|даче)\b"),
        ("on_site", r"\b(?:я|теперь|уже)\s+(?:приехал[аи]?|добрал(?:ся|ась)|приш[её]л|пришла)\s+(?:на участок|на место|на дачу|туда)"),
        ("on_site", r"\bобразец\s+под\s+рукой\b|\bесть\s+(?:у меня\s+)?образец\b"),
        ("on_site", r"\bдержу\s+(?:землю|почву|грунт|образец)\s+в\s+руках\b"),
        ("unknown", r"(?:не знаю|неизвестно).{0,35}(?:доступ|попасть|добраться)"),
        ("unknown", r"\b(?:я\s+)?(?:(?:сейчас|теперь)\s+)?не\s+на\s+(?:участке|месте|даче)\b"),
    )
    offset = 0
    for sentence in sentences(message):
        start = message.find(sentence, offset)
        offset = max(offset, start + len(sentence))
        if "?" in sentence or _HYPOTHETICAL.search(sentence) or _ATTRIBUTED.search(sentence):
            continue
        for status, pattern in patterns:
            for match in re.finditer(pattern, sentence, re.I):
                # Explicitly negated claims must not become positive access facts.
                if re.search(r"\bне\s+$", sentence[:match.start()], re.I):
                    continue
                choices.append((start + match.start(), {"status": status, "quote": sentence}))
    return max(choices, key=lambda value: value[0])[1] if choices else None


def controlled_access_answer(question):
    """A short answer is meaningful only after the program's access question."""
    if not isinstance(question, str):
        return None
    if "?" in question or _ATTRIBUTED.search(question) or _HYPOTHETICAL.search(question):
        return None
    short = _clean(question).strip(" .!")
    choices = (
        ("on_site", r"(?:да|есть|могу|да,? (?:есть|могу))"),
        ("photo_only", r"(?:нет|(?:пока|сейчас) нет|нет доступа|доступа нет)"),
        ("unknown", r"(?:не знаю|не уверен[а]?|неясно|не могу(?: сейчас)?|сейчас не могу|пока не могу)"),
    )
    for status, pattern in choices:
        if re.fullmatch(pattern, short, re.I):
            return {"status": status, "quote": question.strip(), "question_field": "access"}
    return None


# Keep callers of the earlier private parser compatible during the handoff.
_controlled_access_answer = controlled_access_answer


def _separate(rows):
    """Keep visual reports out of fields that imply a longitudinal/tactile test."""
    rows = dict(rows)
    for key in ("water", "texture", "organic_layer"):
        row = rows.get(key)
        if not row or row["status"] == "unknown":
            continue
        quote = row["quote"]
        target = None
        if key == "water" and (not _REGIME.search(quote) or (
            _PHOTO.search(quote) and not re.search(r"обычно|всегда|постоянно|наблюдаю", quote, re.I)
        )):
            target = "surface_water" if _VISIBLE_WATER.search(quote) else None
            rows.pop(key)
        elif key == "texture" and (
            (not _TACTILE.search(quote) and row.get("question") != QUESTIONS["texture"])
            or _PHOTO.search(quote)
        ):
            target = "soil_structure" if _STRUCTURE.search(quote) else (
                "surface_color" if _COLOR.search(quote) else "texture_description"
            )
            rows.pop(key)
        elif key == "organic_layer":
            if _organic_inference(quote):
                target = "organic_description"
                rows.pop(key)
                if _COLOR.search(quote):
                    rows["surface_color"] = dict(row)
            elif not _ORGANIC.search(quote):
                target = "surface_color" if _COLOR.search(quote) else None
                rows.pop(key)
        if target:
            rows[target] = row
    return rows


def record(previous, question, fields=None):
    """Record only latest visitor reports; preserve prior durable user evidence.

    ``previous`` can be this module's result or a case containing ``evidence``.
    An existing program ``pending_field`` binds a brief answer to that one field.
    ``fields`` accepts existing interview rows or extraction-update dictionaries;
    every newly supplied quote is checked against ``question``.
    """
    previous = previous if isinstance(previous, dict) else {}
    memory = _evidence(previous)
    old_fields = _rows(memory.get("fields"))
    question = question if isinstance(question, str) else ""
    pending = previous.get("pending_field", memory.get("pending_field"))
    access_pending = pending == "access"
    pending = pending if isinstance(pending, str) and pending in QUESTIONS else None
    supplied = (
        [{"field": key, **row} for key, row in fields.items() if isinstance(row, dict)]
        if isinstance(fields, dict) else fields if isinstance(fields, list) else []
    )
    updates = []
    for row in supplied:
        if not isinstance(row, dict):
            continue
        quote = _exact_quote(question, row.get("quote"))
        key = row.get("field")
        if not quote or not quote_is_statement(question, quote):
            continue
        if key == "origin" and pending != "origin" and _BRIEF_ORIGIN.fullmatch(_clean(question).strip(" .!")):
            continue
        compact_location = _compact_location_answer(question)
        if key in {"location", "location_detail"} and compact_location and not (
            pending == "location" or compact_location["explicit"]
        ):
            continue
        if key in {"location", "location_detail"} and row.get("status") == "known" and _compact_location_answer(quote):
            # A model cannot turn "не уверен, в Екб" into an unqualified city
            # report merely by extracting the verbatim substring "Екб".
            uncertain = False
            for sentence in sentences(question):
                exact = _exact_quote(sentence, quote)
                if exact:
                    before, after = sentence.split(exact, 1)
                    before = re.sub(r"\b(?:в|во|из|город)\s+$", "", before, flags=re.I)
                    uncertain = uncertain or bool(_REGION_UNCERTAINTY.search(before) or
                        _REGION_UNCERTAINTY.fullmatch(after.strip(" ,:.-–!")))
            if uncertain:
                continue
        if _BARE_UNKNOWN.fullmatch(_clean(quote).strip(" .!")) and key != pending:
            continue
        if any(_ATTRIBUTED.search(sentence) for sentence in sentences(question) if quote in sentence):
            continue
        updates.append({**row, "quote": quote})
    # Existing extraction validates field names, statuses and current-message quotes.
    base = {"fields": old_fields}
    if pending:
        base["pending_field"] = pending
    # The existing extractor limits each batch to five rows; validate each supplied
    # current-message row while retaining earlier country/unknown-town distinctions.
    updated = update_fields(base, question, [])
    for update in updates:
        next_base = {"fields": updated}
        if pending:
            next_base["pending_field"] = pending
        updated = update_fields(next_base, question, [update])
    # Preserve exact original whitespace rather than the extractor's cleaned quote.
    for key, row in list(updated.items()):
        exact = _exact_quote(question, row["quote"])
        if exact:
            related = [sentence for sentence in sentences(question) if _exact_quote(sentence, exact)]
            unsafe = (not quote_is_statement(question, exact) or any(
                _ATTRIBUTED.search(sentence) or (
                    _HYPOTHETICAL.search(sentence) and not _UNKNOWN.search(sentence)
                ) for sentence in related
            ))
            if unsafe:
                if key in old_fields:
                    updated[key] = dict(old_fields[key])
                else:
                    updated.pop(key)
                continue
            row["quote"] = exact
            if key in old_fields and _clean(exact) == _clean(old_fields[key]["quote"]):
                if "question" in old_fields[key]:
                    row["question"] = old_fields[key]["question"]
        elif key in old_fields:
            updated[key] = dict(old_fields[key])
    # Extra report-only keys are intentionally outside the interview's field set.
    updated.update({key: row for key, row in old_fields.items() if key in EXTRA_LABELS})
    organic_report = old_fields.get("organic_layer")
    for sentence in sentences(question):
        if "?" in sentence or _ATTRIBUTED.search(sentence) or (
            _HYPOTHETICAL.search(sentence) and not _UNKNOWN.search(sentence)
        ):
            continue
        unknown = bool(_UNKNOWN.search(sentence))
        row = {"status": "unknown" if unknown else "known", "quote": sentence}
        if len(sentence) > 1000:
            continue
        if _TACTILE.search(sentence) and (_TEXTURE_WORDS.search(sentence) or unknown):
            updated["texture"] = row
        if _REGIME.search(sentence) and not (_PHOTO.search(sentence) and not re.search(
            r"обычно|всегда|постоянно|наблюдаю", sentence, re.I
        )):
            updated["water"] = row
        elif unknown and _WATER_WORDS.search(sentence):
            updated["water"] = row
        if _PHOTO.search(sentence) and _VISIBLE_WATER.search(sentence) and not unknown:
            updated["surface_water"] = row
        if _STRUCTURE.search(sentence) and not unknown and not _HYPOTHETICAL.search(sentence):
            updated["soil_structure"] = row
        if _COLOR.search(sentence) and not unknown and not _HYPOTHETICAL.search(sentence):
            updated["surface_color"] = row
        if _ORGANIC.search(sentence) and not _HYPOTHETICAL.search(sentence):
            if _organic_inference(sentence):
                updated["organic_description"] = row
                # Even a verbatim extractor quote such as "много гумуса" must
                # retain the inference provenance of its complete sentence.
                if organic_report:
                    updated["organic_layer"] = dict(organic_report)
                else:
                    updated.pop("organic_layer", None)
            else:
                updated["organic_layer"] = row
                organic_report = row
        regions = list(_REGION.finditer(sentence))
        uncertain_region = regions and (
            _REGION_UNCERTAINTY.search(sentence[:regions[0].start()])
            or _REGION_UNCERTAINTY.fullmatch(sentence[regions[-1].end():].strip(" ,:.-–!"))
        )
        if uncertain_region:
            # A short extracted name must not discard the visitor's qualifier,
            # including a choice between two regions they cannot distinguish.
            previous_location = old_fields.get("location", {})
            if previous_location.get("status") == "known":
                updated["location"] = dict(previous_location)
                updated["location_detail"] = {"status": "unknown", "quote": sentence}
            else:
                updated["location"] = {"status": "unknown", "quote": sentence}
        for match in regions if not uncertain_region else []:
            if not re.search(r"\bне\s*$", sentence[:match.start()], re.I) and not re.search(
                r"точн|какую|какая|не знаю", match.group(), re.I
            ) and not _HYPOTHETICAL.search(sentence):
                updated["location"] = {"status": "known", "quote": match.group()}
        retraction = _LOCATION_RETRACTION.search(sentence)
        if _LOCATION_UNKNOWN.search(sentence) or retraction:
            location = updated.get("location", {})
            if retraction and location.get("status") != "known":
                # An extractor may label the unknown region as location; it
                # cannot revoke a broader country the visitor has not retracted.
                previous_location = old_fields.get("location", {})
                if previous_location.get("status") == "known":
                    location = previous_location
            retract_region = retraction and re.search(
                r"\b(?:област\w*|кра[йея]\w*|республик\w*)", location.get("quote", ""), re.I
            )
            key = "location_detail" if location.get("status") == "known" and not retract_region else "location"
            if key == "location_detail":
                updated["location"] = dict(location)
            updated[key] = {"status": "unknown", "quote": sentence}
    short = _clean(question).strip(" .!")
    if pending == "origin" and short and len(short) <= 120 and "?" not in short and not _ATTRIBUTED.search(short):
        if _UNKNOWN.search(short) and re.search(r"\b(?:местн|привозн|завозн)", short, re.I):
            updated["origin"] = {"status": "unknown", "quote": question.strip(), "question": QUESTIONS["origin"]}
        elif _BRIEF_ORIGIN.fullmatch(short):
            updated["origin"] = {"status": "known", "quote": question.strip(), "question": QUESTIONS["origin"]}
    location_answer = _compact_location_answer(question)
    if location_answer and (pending == "location" or location_answer["explicit"]):
        # Keep an already reported region/country and store the short town answer
        # separately. Its verbatim words remain the source, never a guessed region.
        key = "location_detail" if old_fields.get("location", {}).get("status") == "known" else "location"
        updated[key] = {"status": "known", "quote": question.strip()}
        if key == "location_detail":
            updated["location"] = dict(old_fields["location"])
        if pending == "location":
            updated[key]["question"] = QUESTIONS["location"]
    if pending and short and len(short) <= 120 and "?" not in short and (
        _BARE_UNKNOWN.fullmatch(short) or direct_answer(pending, short)
    ) and not _ATTRIBUTED.search(short) and not _HYPOTHETICAL.search(short):
        if _BARE_UNKNOWN.fullmatch(short):
            updated[pending] = {"status": "unknown", "quote": question.strip()}
        if pending in updated and _exact_quote(question, updated[pending]["quote"]):
            updated[pending]["question"] = QUESTIONS[pending]
    old_access = memory.get("access")
    access = (dict(old_access) if isinstance(old_access, dict)
              and isinstance(old_access.get("status"), str)
              and old_access["status"] in ACCESS_LABELS
              and isinstance(old_access.get("quote"), str)
              else {"status": "unknown", "quote": ""})
    access_answer = controlled_access_answer(question) if access_pending else None
    return {"kind": "user_soil_evidence", "version": 1,
            "access": _access_update(question) or access_answer or access,
            "fields": _separate(_rows(updated))}


def context(value):
    """Compact Russian model context; uncertainty does not forbid hypotheses."""
    memory = _evidence(value)
    fields = _separate(_rows(memory.get("fields")))
    access = memory.get("access", {})
    access = access if isinstance(access, dict) else {}
    status = access.get("status")
    lines = ["Память наблюдений: только сообщения посетителя; это сообщения, а не проверенные измерения."]
    access_text = ACCESS_LABELS.get(status, ACCESS_LABELS["unknown"]) if isinstance(status, str) else ACCESS_LABELS["unknown"]
    quote = access.get("quote")
    lines.append("Доступ: " + access_text + (f' — «{quote}»' if isinstance(quote, str) and quote else "."))
    for key in ("location", "water", "surface_water", "texture", "organic_layer"):
        if key not in fields:
            lines.append(f"{FIELD_LABELS[key]}: неизвестно, посетитель не сообщал.")
    for key, row in fields.items():
        state = "неизвестно со слов посетителя" if row["status"] == "unknown" else "сообщено посетителем"
        lines.append(f'{FIELD_LABELS[key]}: {state} — «{row["quote"]}»')
        if row.get("normalized"):
            lines.append(f'Расшифровка сокращения посетителя: {row["normalized"]}; регион не определён по сокращению.')
    lines.append(
        "Вода на одном фото не устанавливает постоянное переувлажнение; неизвестный водный режим остаётся неизвестным. "
        "Механический состав на ощупь не измеряется по фото, комочкам или цвету. "
        "Тёмный цвет не измеряет содержание органического вещества; сообщения об остатках или торфе не заменяют анализ. "
        "Используй эту память для осторожных предварительных гипотез и явно сохраняй неизвестные свойства."
    )
    return "\n".join(lines)


evidence_context = context
