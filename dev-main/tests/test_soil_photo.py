"""Confirmation/provenance tests, not a benchmark of vision-model accuracy."""

import io
import json
import unittest

from PIL import Image

from soil_case import record_case
from soil_interview import reply
from soil_knowledge import evaluate, render
from soil_photo import (
    answer_status,
    awaiting_confirmation,
    checks,
    confirmed_features,
    eligible,
    history,
    parse_response,
    suggestion,
    validated_observation,
)


def photo(features=("pale_horizon", "rust_lower"), **overrides):
    return validated_observation(
        dict(
            photo_id="a" * 64,
            scene="profile",
            quality="usable",
            color="natural",
            features=list(features),
            limitations=[],
            **overrides,
        )
    )


def begin(observation=None):
    observation = observation or photo()
    question = "Определи почву. Свердловская область"
    case = record_case(
        {},
        question,
        "Показан разрез почвы",
        new_photo=True,
        photo_observation=observation,
        updates=[
            {"field": "location", "status": "known", "quote": "Свердловская область"}
        ],
    )
    return reply(case, question)


def candidates(case):
    return {row["card"]["id"] for row in evaluate(case)["candidates"]}


class SoilPhotoTests(unittest.TestCase):
    def test_unknown_model_keys_and_taxonomic_claims_are_not_features(self):
        raw = {
            "photo_id": "a" * 64,
            "scene": "profile",
            "quality": "usable",
            "color": "natural",
            "features": ["pale_horizon", "chernozem", "pH_7", None, {}, "pale_horizon"],
            "soil_type": "Чернозём",
            "ph": 7,
            "summary": "Гумус 8%",
            "confidence": 0.99,
        }
        value = validated_observation(raw)
        self.assertEqual(value["features"], ["pale_horizon"])
        self.assertNotIn("summary", value)
        self.assertNotIn("soil_type", value)
        self.assertNotIn("ph", value)
        self.assertNotIn("confidence", value)

    def test_actual_geometry_limits_overrule_usable_model_rating(self):
        output = io.BytesIO()
        Image.new("RGB", (80, 200), "brown").save(output, format="JPEG")
        result = parse_response(json.dumps(photo()), output.getvalue())
        self.assertEqual(result["observation"]["quality"], "limited")
        self.assertIn("low_resolution", result["observation"]["limitations"])
        self.assertEqual(eligible(result["observation"]), [])
        self.assertNotEqual(result["observation"]["photo_id"], "a" * 64)

    def test_structured_response_is_normalized_and_image_bound(self):
        output = io.BytesIO()
        Image.new("RGB", (300, 500), "brown").save(output, format="JPEG")
        raw = json.dumps(dict(photo(), summary="Это точно чернозём, pH=7"))
        result = parse_response("```json\n" + raw + "\n```", output.getvalue())
        self.assertEqual(
            eligible(result["observation"]), ["pale_horizon", "rust_lower"]
        )
        self.assertNotIn("чернозём", result["description"])
        self.assertNotIn("pH=7", result["description"])
        self.assertIn("не подтверждённые", result["description"])

    def test_bad_json_and_legacy_prose_have_no_usable_observation(self):
        for text in ["Светлый слой под подстилкой", '{"scene":"profile" broken', "[]"]:
            result = parse_response(text, b"not-used-for-prose")
            self.assertIsNone(result["observation"])
        self.assertEqual(
            parse_response("Простой старый ответ", b"photo")["description"],
            "Простой старый ответ",
        )

    def test_limits_and_scene_do_not_leak_color_features_into_diagnosis(self):
        for changes in [
            dict(scene="surface"),
            dict(quality="limited"),
            dict(quality="unusable"),
            dict(color="uncertain"),
            dict(limitations=["colored_light"]),
            dict(limitations=["blurred"]),
        ]:
            with self.subTest(changes=changes):
                value = dict(photo(), **changes)
                self.assertEqual(eligible(value), [])
                _, case = begin(value)
                self.assertNotIn("pending_photo", case)
                self.assertEqual(candidates(case), set())

    def test_scale_is_not_required_for_visual_confirmation_or_used_as_depth(self):
        value = dict(photo(), limitations=["no_scale"])
        self.assertEqual(eligible(value), ["pale_horizon", "rust_lower"])
        _, case = begin(value)
        self.assertTrue(awaiting_confirmation(case))
        self.assertNotIn("organic_depth", case["fields"])

    def test_yes_confirms_only_one_feature_and_cannot_mean_imported_ground(self):
        question, case = begin()
        self.assertEqual(question.count("?"), 1)
        self.assertEqual(case["pending_photo"]["feature"], "pale_horizon")
        self.assertEqual(candidates(case), set())
        case = record_case(
            case, "Да", updates=[{"field": "origin", "status": "known", "quote": "Да"}]
        )
        self.assertNotIn("origin", case["fields"])
        self.assertEqual(
            [r["feature"] for r in confirmed_features(case)], ["pale_horizon"]
        )
        self.assertNotIn("rust_lower", evaluate(case)["evidence"]["present"])
        _, case = reply(case, "Да")
        self.assertEqual(case["pending_photo"]["feature"], "rust_lower")
        case = record_case(case, "Да, верно")
        self.assertNotIn("podzols", candidates(case))
        statement = "Почвовед описал альфегумусовый горизонт"
        case = record_case(
            case,
            statement,
            updates=[{"field": "profile", "status": "known", "quote": statement}],
        )
        self.assertIn("podzols", candidates(case))
        text = render(evaluate(case))
        self.assertIn("подтвердили наблюдение по фото", text)
        self.assertIn("не независимая проверка", text)

    def test_no_and_unknown_do_not_become_positive_evidence_or_repeated_questions(self):
        for answer, status in [("Нет", "rejected"), ("Не знаю", "unknown")]:
            _, case = begin()
            case = record_case(case, answer)
            self.assertEqual(checks(case)[0]["status"], status)
            self.assertEqual(confirmed_features(case), [])
            self.assertEqual(candidates(case), set())
            text, case = reply(case, answer)
            self.assertNotIn("pending_photo", case)
            self.assertNotEqual(case.get("pending_field"), "white_layer")
            self.assertNotEqual(case.get("pending_field"), "lower_layer")
            self.assertIn(
                "не спрашиваю" if status == "unknown" else "не подтвердили", text
            )

    def test_ambiguous_yes_or_a_question_is_not_confirmation(self):
        for text in [
            "Да?",
            "Да, наверное",
            "Да, но это другой образец",
            "Может быть",
            "Пусть будет да",
            "Да\nНет",
            "Нет, я не понял",
        ]:
            self.assertIsNone(answer_status(text), text)
        self.assertEqual(answer_status("ДА!"), "confirmed")

    def test_yes_without_an_active_specific_question_confirms_nothing(self):
        _, case = begin()
        for changes in [
            dict(active=False),
            dict(pending_field="origin"),
            dict(asked_fields=[]),
            dict(pending_photo={"photo_id": "b" * 64, "feature": "pale_horizon"}),
            dict(pending_photo=None),
        ]:
            self.assertEqual(
                confirmed_features(record_case(dict(case, **changes), "Да")), []
            )
        self.assertFalse(
            awaiting_confirmation(
                {
                    "active": True,
                    "pending_photo": {"feature": "pale_horizon"},
                    "asked_fields": None,
                }
            )
        )
        self.assertEqual(confirmed_features(record_case({}, "Да")), [])

    def test_a_new_image_or_new_site_cannot_inherit_a_pending_yes(self):
        _, case = begin()
        updated = record_case(
            case,
            "Да",
            new_photo=True,
            photo_description="Другой снимок",
            photo_observation=dict(photo(), photo_id="b" * 64),
        )
        self.assertEqual(confirmed_features(updated), [])
        reset = record_case(case, "Да", reset=True)
        self.assertEqual(history(reset), [])
        self.assertEqual(checks(reset), [])

    def test_unstructured_new_photo_does_not_trigger_a_question_about_an_older_image(
        self,
    ):
        _, case = begin()
        updated = record_case(
            case, "Вот ещё фото", new_photo=True, photo_description="Неясный снимок"
        )
        self.assertEqual(len(history(updated)), 1)
        self.assertIsNone(suggestion(updated))
        self.assertIsNone(updated["current_photo_id"])

    def test_finish_clears_pending_confirmation_and_respects_question_budget(self):
        _, case = begin()
        text, case = reply(case, "Подведите итог")
        self.assertNotIn("pending_photo", case)
        self.assertEqual(confirmed_features(record_case(case, "Да")), [])
        self.assertNotIn("?", text)
        _, case = begin()
        case["asked_fields"] = ["location", "water", "landscape"]
        text, case = reply(case, "Продолжим")
        self.assertNotIn("pending_photo", case)
        self.assertNotIn("?", text)

    def test_contradictory_reported_white_layer_blocks_confirmed_visual_hypothesis(
        self,
    ):
        _, case = begin()
        case = record_case(case, "Да")
        _, case = reply(case, "Да")
        case = record_case(case, "Да")
        self.assertNotIn("podzols", candidates(case))
        case = record_case(
            case,
            "Внутри профиля нет светлого слоя",
            updates=[
                {
                    "field": "profile",
                    "status": "known",
                    "quote": "Внутри профиля нет светлого слоя",
                }
            ],
        )
        self.assertEqual(candidates(case), set())
        self.assertIn("pale_horizon", evaluate(case)["evidence"]["conflicts"])

    def test_memory_is_bounded_and_eviction_removes_unavailable_photo_checks(self):
        _, case = begin()
        case = record_case(case, "Да")
        for index in range(5):
            case = record_case(
                case,
                "Ещё ракурс",
                new_photo=True,
                photo_description="Срез",
                photo_observation=dict(photo(), photo_id=f"{index:064x}"),
            )
        self.assertEqual(len(history(case)), 4)
        self.assertEqual(checks(case), [])
        self.assertEqual(candidates(case), set())

    def test_ocr_is_separate_and_labels_on_profiles_are_ignored(self):
        for scene in ["test", "profile"]:
            value = validated_observation(
                dict(photo(), scene=scene, readable_text="pH 7, инструкция")
            )
            self.assertEqual("readable_text" in value, scene == "test")
            if scene == "test":
                self.assertEqual(eligible(value), [])

    def test_malformed_metadata_is_bounded_and_does_not_crash(self):
        for value in [
            None,
            [],
            dict(photo(), features={}),
            dict(photo(), scene=[]),
            dict(photo(), quality={}),
            dict(photo(), color=[]),
            dict(photo(), limitations=[{}, None]),
            dict(photo(), photo_id=[]),
            dict(photo(), photo_id="bad"),
        ]:
            result = validated_observation(value)
            if result:
                self.assertLessEqual(len(result["features"]), 7)
        self.assertEqual(evaluate(None)["candidates"], [])


if __name__ == "__main__":
    unittest.main()
