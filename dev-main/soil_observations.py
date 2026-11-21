"""Conservative normalization of visitor reports, independent of model wording.

Every retained quote comes from a visitor message. These are reported properties,
not independently verified measurements. Scene descriptions and assistant answers
are never parsed here.
"""

import re

CONTEXT_FEATURES = {
    "light_band",
    "pale_horizon",
    "rust_lower",
    "surface_crust",
    "dark_surface",
    "plant_remains",
    "water_visible",
}
UNCERTAIN = re.compile(
    r"не знаю|не уверен|неизвест|наверн|возможно|кажется|похож|предполож|\?", re.I
)
NEGATIVE = re.compile(r"\bне\b|\bнет\b|отсутств|\bбез\b", re.I)
NO_LABS = re.compile(
    r"анализ(?:а|ов) (?:у меня |пока |ещ[её] )?нет|нет (?:никаких )?анализ(?:а|ов)|анализ\w* (?:не делал|не сдавал)|без анализ(?:а|ов)",
    re.I,
)
TEXTURE = re.compile(
    r"пес(?:ок|чан|чин)|суглин|супес|глин|липк|лент|шарик|рассып", re.I
)
WATER_REGIME = re.compile(
    r"долго.{0,15}(?:стоит|заста)|после дожд|быстро.{0,15}(уход|впит)|(?:стоит|застаивается).{0,20}(?:недел|месяц|долго)|(?:долго|постоянно|обычно|всегда).{0,20}(?:вод|сыр|мокр|сух)|не застаивается|не уходит|не просыхает",
    re.I,
)
# These names preserve explicit place reports only; they never choose a soil.
COUNTRIES = r"Росси[яию]|Канад[аыуе]|Кита[йяе]|Эфиопи[яию]|Украин[аеуы]|Нидерланд(?:ы|ов|ах)|Намиби[яию]|Франци[яию]|Германи[яию]|Беларус[ьи]|Казахстан(?:а|е)?|Армени[яию]|Грузи[яию]|Испани[яию]|Итали[яию]|Норвеги[яию]|Швеци[яию]|Финлянди[яию]|Польш[аеуы]|США|Бразили[яию]|Инди[яию]|Австрали[яию]|Япони[яию]|Турци[яию]|Узбекистан(?:а|е)?"
PLACE = re.compile(
    r"(?:(?:из|в|это|страна)\s+(?:"
    + COUNTRIES
    + r")\b|\b(?:"
    + COUNTRIES
    + r")\b\s*$)",
    re.I,
)


def clean(text):
    return " ".join(text.split())


def row(quote, status="known"):
    return {"status": status, "quote": clean(quote)[:300]}


def sentences(text):
    return [
        part.strip() for part in re.findall(r"[^.!?;\n]+[.!?;]?", text) if part.strip()
    ]


def quote_is_statement(question, quote):
    normalized = clean(question).casefold()
    spans = list(re.finditer(r"[^.!?;\n]+[.!?;]?", normalized))
    for occurrence in re.finditer(re.escape(clean(quote).casefold()), normalized):
        overlapping = [
            m.group()
            for m in spans
            if m.start() < occurrence.end() and m.end() > occurrence.start()
        ]
        if overlapping and all("?" not in text for text in overlapping):
            return True
    return False


def normalize_fields(values):
    """Repair semantic field assignments, including older saved cards."""
    result = (
        {
            key: dict(value)
            for key, value in values.items()
            if isinstance(value, dict)
            and isinstance(value.get("quote"), str)
            and isinstance(value.get("status"), str)
            and value["status"] in {"known", "unknown"}
        }
        if isinstance(values, dict)
        else {}
    )
    texture = result.get("texture", {})
    quote = texture.get("quote", "")
    if texture.get("status") == "known" and not TEXTURE.search(quote):
        result.pop("texture", None)
        if re.search(r"комоч|кусоч|глыб|трещ|столб", quote, re.I):
            result.setdefault("soil_structure", texture)
        elif re.search(r"камн", quote, re.I):
            result.setdefault("stones", texture)
    water = result.get("water", {})
    if water.get("status") == "known" and not WATER_REGIME.search(
        water.get("quote", "")
    ):
        result.pop("water", None)
        if re.search(r"вод|мокр|луж", water.get("quote", ""), re.I):
            result.setdefault("surface_water", water)
    landscape = result.get("landscape", {})
    if (
        landscape.get("status") == "known"
        and re.search(r"трав|корн", landscape.get("quote", ""), re.I)
        and not re.search(
            r"лес|луг|поле|сад|степь|болот|пойм", landscape.get("quote", ""), re.I
        )
    ):
        result.pop("landscape", None)
        result.setdefault("vegetation", landscape)
    return result


def enrich_fields(question, incoming, previous):
    result = normalize_fields(incoming)
    for sentence in sentences(question):
        if "?" in sentence:
            continue
        place = PLACE.search(sentence)
        if (
            place
            and not re.search(
                r"наверн|возможно|кажется|предполож", sentence[: place.end()], re.I
            )
            and not re.search(r"\bне\s*$", sentence[: place.start()], re.I)
            and not UNCERTAIN.search(
                sentence[
                    max(
                        sentence.rfind(",", 0, place.start()),
                        sentence.rfind(":", 0, place.start()),
                    )
                    + 1 : place.end()
                ]
            )
            and not re.match(r"\s+ли\b", sentence[place.end() :], re.I)
        ):
            result["location"] = row(place.group())
        if re.search(r"област|край|республик", sentence, re.I):
            # An existing model-extracted known region must survive an unknown town.
            known = incoming.get("location", {})
            if known.get("status") == "known":
                result["location"] = known
        if UNCERTAIN.search(sentence) or re.search(
            r"не (?:проверял|измерял|могу|делал)", sentence, re.I
        ):
            if re.search(
                r"на ощупь|в руках|тестов руками|пощуп|шарик|лент|механическ|песчин",
                sentence,
                re.I,
            ):
                result.setdefault("texture", row(sentence, "unknown"))
            if re.search(
                r"влажност|водный режим|мокро ли|после дожд|\bвод[ауы]\b",
                sentence,
                re.I,
            ):
                result.setdefault("water", row(sentence, "unknown"))
            if re.search(
                r"растительност|окружени|что (?:вокруг|дальше)|за кадром",
                sentence,
                re.I,
            ):
                result.setdefault("landscape", row(sentence, "unknown"))
        elif WATER_REGIME.search(sentence):
            result["water"] = row(sentence)
        if re.search(
            r"(?:видна|видно|сверху|вокруг).{0,25}(?:сухая )?трава", sentence, re.I
        ) and not re.search(r"видна ли|трава\s+не|нет.{0,10}трав", sentence, re.I):
            fragment = re.search(
                r"(?:видна|видно|сверху|вокруг).{0,25}(?:сухая )?трава", sentence, re.I
            )
            if not UNCERTAIN.search(sentence[: fragment.end()]):
                result["vegetation"] = row(fragment.group())
        if NO_LABS.search(sentence):
            result["tests"] = row(NO_LABS.search(sentence).group(), "unknown")
    old_location = previous.get("location", {})
    location = result.get("location", {})
    if (
        location.get("status") == "unknown"
        and old_location.get("status") == "known"
        and re.search(r"город|точн|насел|место", location.get("quote", ""), re.I)
    ):
        result["location_detail"] = location
        result["location"] = old_location
    # A known country/region and an unknown town are different facts.
    if result.get("location", {}).get("status") == "known":
        for sentence in sentences(question):
            if UNCERTAIN.search(sentence) and re.search(
                r"точн|город|насел", sentence, re.I
            ):
                detail = re.search(r"(?:точн\w*|город|насел)[^.!?;]*", sentence, re.I)
                if detail:
                    result["location_detail"] = row(detail.group(), "unknown")
    return result


def record_features(previous, question):
    result = {
        (item["feature"], item["status"]): dict(item)
        for item in feature_rows({"reported_features": previous})
    }
    for sentence in sentences(question):
        if UNCERTAIN.search(sentence) or len(sentence) > 300:
            continue
        text = sentence.casefold().replace("ё", "е")
        negative = bool(NEGATIVE.search(text))

        def add(feature):
            status = "absent" if negative else "present"
            result[feature, status] = {
                "feature": feature,
                "status": status,
                "quote": clean(sentence),
                "source": "user_report",
            }

        light = re.search(
            r"(?:светл|белес|бел[а-я]*).{0,20}(?:сло|полос|горизонт|прослой)|(?:сло|полос|горизонт|прослой).{0,20}(?:светл|белес|бел[а-я]*)",
            text,
        )
        stacked = re.search(
            r"(?:сверху|вверху).{0,35}(?:темн|черн).{0,35}под (?:ней|ним).{0,12}(?:светл|бел)",
            text,
        )
        gradient = re.search(
            r"(?:сверху|вверху).{0,35}(?:темн|черн).{0,35}(?:ниже|дальше).{0,15}светл",
            text,
        )
        if light or stacked or gradient:
            add("light_band")
            if stacked or re.search(
                r"внутри|в разрезе|в профиле|под (?:темн|черн|подстил)", text
            ):
                add("pale_horizon")
            if re.search(r"(?:под|ниже).{0,30}(?:рыж|бур|охр)", text):
                add("rust_lower")
        if (
            re.search(r"поверхност", text)
            and re.search(r"бел|солев", text)
            and re.search(r"налет|корк", text)
        ):
            add("surface_crust")
        if re.search(r"темн|черн", text) and re.search(r"земл|сло|полос", text):
            add("dark_surface")
        if re.search(r"корн|растительн.{0,15}остат", text):
            add("plant_remains")
        if re.search(
            r"(?:видна|видно).{0,20}вода|вода.{0,20}(?:видна|на снимке)|в (?:выемке|яме).{0,20}вод",
            text,
        ):
            add("water_visible")
    return list(result.values())[-32:]


def feature_rows(case):
    values = case.get("reported_features", []) if isinstance(case, dict) else []
    return (
        [
            dict(item)
            for item in values[-32:]
            if isinstance(item, dict)
            and isinstance(item.get("feature"), str)
            and item["feature"] in CONTEXT_FEATURES
            and isinstance(item.get("status"), str)
            and item["status"] in {"present", "absent"}
            and item.get("source") == "user_report"
            and isinstance(item.get("quote"), str)
            and 2 <= len(item["quote"]) <= 300
        ]
        if isinstance(values, list)
        else []
    )


def lab_questions_available(fields):
    tests = fields.get("tests", {})
    return (
        tests.get("status") == "known"
        and bool(re.search(r"анализ|лаборатор", tests.get("quote", ""), re.I))
        and not NO_LABS.search(tests.get("quote", ""))
    )
