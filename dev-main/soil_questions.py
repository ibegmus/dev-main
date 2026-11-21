"""Questions selected by unresolved meaning and available observations."""

import re
from soil_observations import TEXTURE, lab_questions_available, normalize_fields
from soil_photo import history, suggestion

LAB_FIELDS = {"salinity", "sodicity", "carbonate"}


def current_photo(case):
    return next(
        (p for p in history(case) if p["photo_id"] == case.get("current_photo_id")),
        None,
    )


def blocked(case, field, assessment):
    fields = normalize_fields(case.get("fields", {}))
    asked = case.get("asked_fields", [])
    entry = fields.get(field, {})
    present = assessment["evidence"]["present"]
    if field in asked or entry.get("status") == "unknown":
        return True
    known = entry.get("status") == "known"
    if field in LAB_FIELDS:
        if known:
            return True
        return not lab_questions_available(fields)
    if field == "white_layer":
        return any(f in present for f in ["pale_horizon", "surface_crust"])
    if field == "lower_layer":
        return (
            any(f in present for f in ["rust_lower", "clay_increase"])
            or known
            and bool(TEXTURE.search(entry.get("quote", "")))
        )
    if field == "texture":
        return known or any(f in present for f in ["sandy_material", "loamy_material"])
    if field == "organic_depth":
        return known or "peat_thickness" in present
    if field == "water":
        return (
            known
            or "long_waterlogging" in present
            or "long_waterlogging" in assessment["evidence"]["absent"]
        )
    if field == "organic_layer":
        return any(
            f in present
            for f in [
                "organic_material",
                "mineral_material",
                "mineral_humus",
                "gray_humus",
                "dark_humus",
            ]
        )
    if field == "profile":
        photo = current_photo(case)
        return known or bool(
            photo and photo["scene"] == "profile" and photo["quality"] == "usable"
        )
    return known


def photo_question(case, assessment, field=None):
    present = assessment["evidence"]["present"]
    allowed = {
        "pale_horizon",
        "rust_lower",
        "gley_colors",
        "redox_mottles",
        "surface_crust",
    }
    if "dark_humus" in present:
        allowed.add("granular_structure")
    if "loamy_material" in present:
        allowed.add("brown_structured")
    available = {
        "location",
        "landscape",
        "water",
        "origin",
        "white_layer",
        "lower_layer",
        "organic_layer",
        "texture",
        "soil_structure",
        "profile",
        "redox",
        "clay_change",
        "organic_depth",
        *LAB_FIELDS,
    }
    return suggestion(
        case,
        field,
        blocked_fields={f for f in available if blocked(case, f, assessment)},
        allowed_features=allowed,
        supported_features=set(present),
    )


def choose(case, assessment, photo_description=""):
    present = assessment["evidence"]["present"]
    if assessment["evidence"]["imported"] or assessment["evidence"]["conflicts"]:
        return None

    def first(fields):
        return next((f for f in fields if not blocked(case, f, assessment)), None)

    if not blocked(case, "location", assessment):
        return "location"
    photo = current_photo(case)
    if photo and photo["quality"] != "usable":
        return first(["profile"])
    if any(f in present for f in ["light_band", "pale_horizon", "surface_crust"]):
        field = first(["white_layer"])
        if field:
            return field
        if "pale_horizon" in present:
            field = first(["lower_layer", "texture"])
            if field:
                return field
            if "loamy_material" in present:
                field = first(["clay_change"])
                if field:
                    return field
    if any(f in present for f in ["water_visible", "gley_colors", "long_waterlogging"]):
        field = first(["water", "redox"])
        if field:
            return field
    if any(f in present for f in ["dark_surface", "organic_material"]):
        field = first(["organic_layer"])
        if field:
            return field
        if "organic_material" in present:
            field = first(["organic_depth", "water"])
            if field:
                return field
    if "exchangeable_sodium" in present:
        field = first(["soil_structure"])
        if field:
            return field
    if "gray_humus" in present:
        field = first(["clay_change"])
        if field:
            return field
    if any(f in present for f in ["dark_humus", "gray_humus", "brown_structured"]):
        field = first(
            [
                "origin",
                "texture",
                "clay_change" if "gray_humus" in present else "soil_structure",
            ]
        )
        if field:
            return field
    visual = photo_question(case, assessment)
    if visual:
        return visual["field"]
    # Legacy photo prose can suggest a question, never a measured feature.
    priorities = []
    if not photo:
        if re.search(r"бел|светл", photo_description, re.I):
            priorities.append("white_layer")
        if re.search(r"вода|влаж|мокр", photo_description, re.I):
            priorities.append("water")
        if re.search(r"торф|органич|ч[её]рн", photo_description, re.I):
            priorities.append("organic_layer")
    priorities += ["origin", "texture", "water", "landscape"]
    return first(priorities)


def final_step(case, assessment):
    photo = current_photo(case)
    if photo and photo["quality"] != "usable":
        return "Следующий полезный материал — более чёткий снимок уже доступного среза при дневном свете с масштабом. Текущее изображение ограничено по качеству; глубокую яму копать не нужно."
    present = assessment["evidence"]["present"]
    if any(f in present for f in ["pale_horizon", "light_band", "surface_crust"]):
        return "Для следующей проверки полезнее установить состав светлого материала и различия соседних слоёв. Положение уже описанного или подтверждённого слоя учтено; сохранённые наблюдения можно передать почвоведу вместе с готовыми анализами, если они появятся."
    if "water_visible" in present and "long_waterlogging" not in present:
        return "Сейчас не хватает сведений об обычном водном режиме: разовый мокрый снимок его не показывает. Если появятся наблюдения после дождей, их можно добавить к сохранённому описанию."
    if any(f in present for f in ["dark_surface", "organic_material"]):
        return "Для следующей проверки важны состав и мощность тёмного слоя: преобладает ли растительный материал или минеральная земля. Готовые сведения можно добавить позже; цвет не заменяет эту проверку."
    if photo and photo["scene"] == "profile":
        return "Фото разреза уже учтено. Сохранённое описание можно передать почвоведу; новые наблюдения или готовые результаты анализов можно добавить в эту же беседу."
    return "Наблюдения сохранены. Для дальнейшей проверки полезен уже доступный срез с видимой сменой слоёв и масштабом; глубокую яму копать не нужно."
