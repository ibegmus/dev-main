"""Source-linked candidate directions, never an automatic taxonomic determination.

Evidence consists of visitor quotes, normalized reports and explicit confirmations.
Image predictions alone and past model answers are not facts. Counts of matching
clues are not probabilities.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from soil_observations import feature_rows, record_features, normalize_fields

DATA = Path(__file__).resolve().parent / "reference_data/soil_rules.json"
FEATURES = {
    "light_band": (
        "сообщённая светлая полоса, положение требует уточнения",
        "white_layer",
    ),
    "dark_surface": ("сообщённая тёмная окраска земли", "organic_layer"),
    "plant_remains": ("видимые корни или растительные остатки", "organic_layer"),
    "water_visible": ("вода видна на снимке", "water"),
    "alfhumic_horizon": (
        "альфегумусовый горизонт указан в описании специалиста",
        "profile",
    ),
    "pale_horizon": ("осветлённый слой внутри профиля", "white_layer"),
    "rust_lower": ("рыжевато-бурый слой ниже осветлённого", "lower_layer"),
    "sandy_material": ("песчаный материал", "texture"),
    "clay_increase": ("увеличение глинистости вниз по профилю", "lower_layer"),
    "clay_films": ("описанные глинистые плёнки на гранях отдельностей", "clay_change"),
    "mineral_humus": ("обособленный минеральный гумусовый горизонт", "organic_layer"),
    "gray_humus": ("описанный серогумусовый горизонт", "organic_layer"),
    "dark_humus": ("описанный тёмный минеральный гумусовый горизонт", "organic_layer"),
    "granular_structure": ("описанная зернистая структура", "soil_structure"),
    "carbonate_below": (
        "подтверждённые карбонаты в нижележащем горизонте",
        "carbonate",
    ),
    "gley_colors": ("сизо-голубая или серо-зелёная окраска", "redox"),
    "redox_mottles": ("ржавые пятна в сизой минеральной массе", "redox"),
    "long_waterlogging": ("длительное насыщение водой", "water"),
    "organic_material": (
        "толща растительного органического материала",
        "organic_layer",
    ),
    "peat_thickness": ("известна мощность именно торфяной толщи", "organic_depth"),
    "mineral_material": ("материал описан как минеральный, не торф", "organic_layer"),
    "surface_crust": ("белый налёт или корка на поверхности", "white_layer"),
    "soluble_salts": ("анализ сообщает повышенные водорастворимые соли", "salinity"),
    "columnar_structure": (
        "столбчатая или столбчато-призматическая структура",
        "soil_structure",
    ),
    "exchangeable_sodium": ("анализ сообщает повышенный обменный натрий", "sodicity"),
    "weak_profile": ("подстилка над песком без развитой системы горизонтов", "profile"),
    "brown_structured": (
        "бурый слой с описанной комковатой/ореховатой структурой",
        "soil_structure",
    ),
    "loamy_material": ("суглинистый или глинистый материал", "texture"),
    "weak_clay_transfer": (
        "описано отсутствие выраженного иллювиирования глины",
        "clay_change",
    ),
}
GATES = {
    "podzols": "pale_horizon",
    "podzolic": "pale_horizon",
    "gray": "gray_humus",
    "chernozems": "dark_humus",
    "gleyzems": "gley_colors",
    "peat": "organic_material",
    "solonchaks": "soluble_salts",
    "solonetz": "exchangeable_sodium",
    "psammozems": "sandy_material",
    "burozems": "brown_structured",
}
HEDGE = re.compile(
    r"не знаю|неизвест|наверн|возможно|кажется|предполож|похож|не уверен|если|\?", re.I
)
NEGATIVE = re.compile(r"\bне\b|\bнет\b|отсутств|\bбез\b", re.I)
FORMULA_TRANSLATION = str.maketrans("ОСНЕАВТУМ", "OCHEABTYM")


def has(pattern, text):
    return bool(re.search(pattern, text, re.I))


@lru_cache(maxsize=1)
def reference():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    ids = set()
    for card in data["cards"]:
        if card["id"] in ids:
            raise ValueError("Duplicate soil rule")
        ids.add(card["id"])
        if any(
            f not in FEATURES
            for f in card["required_features"]
            + card["exclude_features"]
            + card.get("mandatory_features", [])
        ):
            raise ValueError("Unknown diagnostic feature")
        if not card["sources"] or any(
            not s["url"].startswith("https://soils.narod.ru/taxon/type/")
            for s in card["sources"]
        ):
            raise ValueError("Missing primary source")
    return data


def observations(case):
    fields = case.get("fields", {}) if isinstance(case, dict) else {}
    if not isinstance(fields, dict):
        return []
    fields = normalize_fields(fields)
    return [
        (key, row["quote"])
        for key, row in fields.items()
        if isinstance(row, dict)
        and row.get("status") == "known"
        and isinstance(row.get("quote"), str)
        and not HEDGE.search(row["quote"])
    ]


def extract_features(case):
    found = {}
    absent = {}
    formulas = []
    imported = False

    def add(name, key, text, target=None):
        entry = {"field": key, "quote": text[:300]}
        (found if target is None else target).setdefault(name, []).append(entry)

    for key, text in observations(case):
        t = text.casefold().replace("ё", "е")
        negative = bool(NEGATIVE.search(t))
        if (
            key == "profile"
            and not negative
            and has(r"специалист|почвовед|лаборатор", t)
            and has(r"альфегумусов|\bBHF\b", t)
        ):
            add("alfhumic_horizon", key, text)
        if (key == "origin" and t.strip(" .!") == "да") or (
            key in {"origin", "profile", "organic_layer"}
            and not negative
            and has(r"привоз|завоз|насыпн|покупн|завез|завезл", t)
        ):
            imported = True
        if (
            key in {"profile", "white_layer", "lower_layer"}
            and has(r"(светл|белес|бел[ыо])", t)
            and has(r"сло|горизонт", t)
            and has(r"под |ниже|внутри|глубин|после|профил", t)
        ):
            add("pale_horizon", key, text, absent if negative else None)
        if (
            key in {"profile", "lower_layer"}
            and not negative
            and has(
                r"(под|ниже).{0,35}(бел|светл).{0,40}(рыж|бур|охр)|(рыж|бур|охр).{0,35}(под|ниже).{0,35}(бел|светл)",
                t,
            )
        ):
            add("rust_lower", key, text)
        if (
            key == "lower_layer"
            and "lower_layer" in case.get("asked_fields", [])
            and not negative
            and has(r"рыж|бур|охр", t)
        ):
            add("rust_lower", key, text)
        if key in {"profile", "clay_change", "lower_layer"} and not negative:
            if has(r"(ниже|внизу|вниз).{0,60}(глинист|глин|тяжел)", t) and has(
                r"более|больше|сильнее|глинистее|глина", t
            ):
                add("clay_increase", key, text)
            if has(r"глинист.{0,12}плен|кутан", t):
                add("clay_films", key, text)
        if key in {"texture", "profile", "lower_layer"} and not negative:
            if has(r"песчан|песок", t) and not has(r"мало пес|песка мало", t):
                add("sandy_material", key, text)
            if has(r"суглин|глинист|глина", t):
                add("loamy_material", key, text)
        if key in {"organic_layer", "profile"} and not negative:
            if has(r"гумусов|гумусн", t):
                if has(r"минеральн", t):
                    add("mineral_humus", key, text)
                if has(r"сер", t) and not has(r"темно-сер", t):
                    add("gray_humus", key, text)
                if has(r"темн|черн", t):
                    add("dark_humus", key, text)
            if has(r"торф", t) or (
                has(r"растительн.{0,12}остат", t)
                and has(r"толща|состоит|преоблада|в основном", t)
            ):
                add("organic_material", key, text)
            if has(r"минеральн", t):
                add("mineral_material", key, text)
        if (
            key == "organic_layer"
            and has(r"минеральн", t)
            and has(r"не торф|нет торф", t)
        ):
            add("mineral_material", key, text)
        if (
            key in {"organic_depth", "organic_layer"}
            and not negative
            and has(r"\d+(?:[.,]\d+)?\s*(?:см|сантиметр|мм|метр|м\b)", t)
        ):
            if has(r"торф", t) or (
                key == "organic_depth" and key in case.get("asked_fields", [])
            ):
                add("peat_thickness", key, text)
        if key in {"water", "profile"}:
            positive = has(
                r"долго.{0,15}(стоит|заста)|(?:стоит|застаивается).{0,20}(недел|месяц|долго)|не уходит|не просыхает|длительн.{0,20}(насыщ|переувлаж)",
                t,
            )
            dry = has(r"не застаивается|луж нет|вода быстро уходит|обычно сухо", t)
            if positive or dry:
                add("long_waterlogging", key, text, absent if dry else None)
        if key in {"redox", "profile"} and not negative:
            if has(r"сиз|голубоват|серо-зелен", t):
                add("gley_colors", key, text)
            if has(r"сиз|голубоват|серо-зелен", t) and has(
                r"ржав.{0,12}пят|рыж.{0,12}пят|охрист.{0,12}пят", t
            ):
                add("redox_mottles", key, text)
        if (
            key in {"white_layer", "salinity", "profile"}
            and not negative
            and has(r"поверхност", t)
            and has(r"бел|солев", t)
            and has(r"корк|налет", t)
        ):
            add("surface_crust", key, text)
        if (
            key in {"tests", "salinity"}
            and not negative
            and has(r"анализ|лаборатор", t)
            and has(r"водорастворим.{0,15}сол|легкорастворим.{0,15}сол", t)
            and has(r"повыш|высок", t)
        ):
            add("soluble_salts", key, text)
        if (
            key in {"tests", "sodicity"}
            and not negative
            and has(r"анализ|лаборатор", t)
            and has(r"обменн.{0,12}натри", t)
            and has(r"повыш|высок", t)
        ):
            add("exchangeable_sodium", key, text)
        if key in {"soil_structure", "profile"} and not negative:
            if has(r"зернист", t):
                add("granular_structure", key, text)
            if has(r"столбчат|вертикальн.{0,10}столбик", t):
                add("columnar_structure", key, text)
            if has(r"бур|коричнев", t) and has(r"комковат|ореховат", t):
                add("brown_structured", key, text)
        if (
            key in {"tests", "carbonate", "profile"}
            and not negative
            and has(r"карбонат", t)
            and has(r"ниже|нижн|под гумус", t)
            and has(r"анализ|специалист|горизонт", t)
        ):
            add("carbonate_below", key, text)
        if key in {"profile", "clay_change"} and has(
            r"(нет|отсутств|слаб).{0,20}иллювиир", t
        ):
            add("weak_clay_transfer", key, text)
        if (
            key == "profile"
            and has(r"подстилк", t)
            and has(r"песок|песчан", t)
            and has(r"нет.{0,20}горизонт|горизонт.{0,20}(не выраж|не различ)|сразу", t)
        ):
            add("weak_profile", key, text)
        if key == "profile" and not negative:
            canonical = (
                text.upper()
                .translate(FORMULA_TRANSLATION)
                .replace("—", "-")
                .replace("–", "-")
                .replace(",", "")
            )
            canonical = re.sub(r"\s*-\s*", "-", canonical)
            for card in reference()["cards"]:
                for formula in card["formulas"]:
                    if re.search(
                        r"(?<![A-ZА-Я-])" + re.escape(formula) + r"(?![A-ZА-Я-])",
                        canonical,
                    ):
                        formulas.append({"card": card["id"], "quote": text[:300]})
    normalized = feature_rows(case)
    for key, quote in observations(case):
        if key in {
            "profile",
            "white_layer",
            "lower_layer",
            "organic_layer",
            "surface_water",
            "vegetation",
        }:
            normalized += record_features([], quote)
    for entry in normalized:
        target = found if entry["status"] == "present" else absent
        feature = entry["feature"]
        evidence = {
            "field": FEATURES[feature][1],
            "source": "user_report",
            "quote": entry["quote"],
        }
        if evidence not in target.setdefault(feature, []):
            target[feature].append(evidence)
    from soil_photo import confirmed_features

    for row in confirmed_features(case):
        found.setdefault(row["feature"], []).append(
            {
                "source": "photo_confirmed",
                "photo_id": row["photo_id"],
                "quote": row["answer"],
            }
        )
    return {
        "present": found,
        "absent": absent,
        "formulas": formulas,
        "imported": imported,
        "conflicts": sorted(set(found) & set(absent)),
    }


def evaluate(case):
    data = reference()
    ev = extract_features(case)
    candidates = []
    partials = []
    if not ev["imported"] and not ev["conflicts"]:
        for card in data["cards"]:
            if any(f in ev["present"] for f in card["exclude_features"]):
                continue
            if any(f in ev["absent"] for f in card["required_features"]):
                continue
            matched = [f for f in card["required_features"] if f in ev["present"]]
            formal = next((x for x in ev["formulas"] if x["card"] == card["id"]), None)
            item = {
                "card": card,
                "matched": matched,
                "missing": [
                    f for f in card["required_features"] if f not in ev["present"]
                ],
                "formula": formal,
            }
            if matched or formal:
                partials.append(item)
            if (
                formal
                and card["id"] == "psammozems"
                and "sandy_material" not in matched
            ):
                formal = None
                item["formula"] = None
            if formal or (
                len(matched) >= card["minimum_features"]
                and GATES[card["id"]] in matched
                and all(
                    feature in matched for feature in card.get("mandatory_features", [])
                )
            ):
                candidates.append(item)
    candidates.sort(
        key=lambda c: (-bool(c["formula"]), -len(c["matched"]), c["card"]["id"])
    )
    return {
        "version": data["version"],
        "evidence": ev,
        "candidates": candidates[:3],
        "partials": partials,
    }


def render_partial(assessment):
    present = assessment["evidence"]["present"]
    if any(f in present for f in ["light_band", "pale_horizon", "surface_crust"]):
        details = [
            "Светлая окраска материала учтена. Её цвет не устанавливает состав материала или тип почвы."
        ]
        if "rust_lower" in present:
            details.append(
                "Рыжевато-бурый материал ниже тоже учтён; сочетание двух цветов само по себе не различает процессы почвообразования."
            )
        details.append(
            "Среди объяснений светлой окраски проверяют вынос окрашивающих соединений и накопление светлых солей или карбонатов. Для различения нужны положение и свойства соседних слоёв, механический состав и, если доступны, анализы состава материала. Проценты вероятности по этим признакам не рассчитываются."
        )
        details.append(
            "Источники для проверки признаков:\nhttps://soils.narod.ru/taxon/type/podz07.html\nhttps://soils.narod.ru/taxon/type/texdipod23.html\nhttps://soils.narod.ru/taxon/type/slabosolon19.html\nhttps://soils.narod.ru/taxon/type/sher06.html"
        )
        return "\n".join(details)
    if any(f in present for f in ["water_visible", "gley_colors", "long_waterlogging"]):
        return "Наблюдения о воде и окраске учтены. Вода на одном снимке не доказывает длительного переувлажнения или глеевого горизонта. Для различения нужны обычный водный режим и описание сизых участков и пятен в профиле.\nИсточник для проверки признаков: https://soils.narod.ru/taxon/type/gl11.html"
    if any(f in present for f in ["dark_surface", "plant_remains", "organic_material"]):
        return "Окраска и растительные остатки учтены. Корни и тёмный цвет бывают и в минеральной земле; они не доказывают ни торфяную толщу, ни чернозём. Для следующего различения нужны состав и мощность слоя, а не повторное подтверждение цвета.\nИсточники для проверки признаков:\nhttps://soils.narod.ru/taxon/type/torfoli24.html\nhttps://soils.narod.ru/taxon/type/sher06.html"
    if "sandy_material" in present:
        return "Песчаный материал учтён как механический состав. Он не задаёт тип почвы: нужна система горизонтов и её диагностические свойства.\nИсточник для проверки признаков: https://soils.narod.ru/taxon/type/slabopsam19.html"
    return "Сочетания сообщённых вами признаков пока недостаточно для справочной гипотезы. Это не исключает типы, отсутствующие в небольшом наборе правил."


def render(assessment):
    ev = assessment["evidence"]
    if ev["imported"]:
        return "Вы сообщили о привозном или насыпном материале. Для него нельзя автоматически назначить природный тип по региону. Сначала нужно отделить привезённый слой от исходной почвы."
    if ev["conflicts"]:
        return (
            "В сообщениях расходятся сведения о признаках: "
            + ", ".join(FEATURES[f][0] for f in ev["conflicts"])
            + ". Пока это не уточнено, справочные гипотезы не выбираются."
        )
    if not assessment["candidates"]:
        return render_partial(assessment)
    lines = [
        "Справочные гипотезы по вашим сообщениям, не установленный тип почвы. Названия — из российского научного справочника, без перевода в WRB. Правила ещё не прошли независимую проверку почвоведом; порядок не означает вероятность."
    ]
    lines.append(
        "Происхождение грунта нужно проверять отдельно: справочные схемы не устанавливают строение привозной смеси."
    )
    for item in assessment["candidates"]:
        card = item["card"]
        lines.append(card["title"])
        for f in item["matched"]:
            sources = ev["present"][f]
            direct = next(
                (x for x in sources if x.get("source") != "photo_confirmed"), None
            )
            if direct:
                lines.append("За: " + FEATURES[f][0] + " — «" + direct["quote"] + "».")
            else:
                lines.append(
                    "За: "
                    + FEATURES[f][0]
                    + " — вы подтвердили наблюдение по фото ответом «"
                    + sources[0]["quote"]
                    + "»; это не независимая проверка."
                )
        if item["formula"]:
            lines.append(
                "За: вы привели формулу профиля «"
                + item["formula"]["quote"]
                + "», совпадающую со справочной схемой; её достоверность независимо не проверена."
            )
        if item["missing"]:
            lines.append(
                "Не хватает: "
                + ", ".join(FEATURES[f][0] for f in item["missing"])
                + "."
            )
        lines.append(card["summary"] + " " + card["limitations"])
        lines.append("Перед определением типа нужны: " + card["confirmation_needed"])
        for source in card["sources"]:
            lines.append("Источник: " + source["url"])
    lines.append(
        "Совпадение с частью признаков не заменяет диагностику всех горизонтов. Список не исчерпывает почвы России; дозы удобрений и точные pH/NPK из этих гипотез не следуют."
    )
    return "\n".join(lines)
