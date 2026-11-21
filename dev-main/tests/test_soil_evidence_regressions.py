"""Live-v3 failure inputs and held-out paraphrases; no model accuracy claim."""

import json
import unittest
from pathlib import Path
from soil_case import record_case
from soil_interview import choose_question, reply, validated_updates
from soil_knowledge import evaluate, render
from soil_photo import validated_observation
from soil_evidence import record as record_evidence, context as evidence_context


def known(**values):
    return {
        "fields": {
            key: {"status": "known", "quote": quote} for key, quote in values.items()
        }
    }


class SoilEvidenceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.loads(
            (Path(__file__).parent / "fixtures/soil_v3_reports.json").read_text()
        )["cases"]

    def replay_second_message(self, index):
        fixture = self.fixtures[index - 1]
        previous = {
            "fields": {},
            "asked_fields": ["location"],
            "pending_field": "location",
        }
        return record_case(
            previous, fixture["question"], updates=fixture["visible_v3_updates"]
        )

    def test_organic_inference_from_photo_keeps_visitor_provenance(self):
        for message in (
            "На фото чёрная почва, значит много гумуса",
            "Тёмная земля, поэтому она богата органическим веществом",
        ):
            for supplied_quote in (None, message, message.split(", ", 1)[1].split(" ", 1)[1]):
                supplied = ({"organic_layer": {"status": "known", "quote": supplied_quote}}
                            if supplied_quote else None)
                with self.subTest(message=message, supplied=supplied):
                    memory = record_evidence({}, message, supplied)
                    self.assertNotIn("organic_layer", memory["fields"])
                    self.assertEqual(memory["fields"]["surface_color"]["quote"], message)
                    self.assertEqual(memory["fields"]["organic_description"]["quote"], message)
                    text = evidence_context(memory)
                    self.assertIn("Гипотеза посетителя", text)
                    self.assertIn(message, text)
                    self.assertIn("Органический материал (сообщение без подтверждения анализом): неизвестно", text)

    def test_organic_reports_are_retained_independently_of_later_color_inference(self):
        for report in (
            "Анализ показал гумус 3%",
            "Вижу волокнистый торф",
            "Почва с растительными остатками",
        ):
            with self.subTest(report=report):
                memory = record_evidence({}, report)
                self.assertEqual(memory["fields"]["organic_layer"], {
                    "status": "known", "quote": report,
                })
                inference = "На фото чёрная почва, значит много гумуса"
                for previous, message in ((memory, inference), ({}, report + ". " + inference)):
                    updated = record_evidence(previous, message)
                    self.assertEqual(updated["fields"]["organic_layer"]["quote"], report + ("." if not previous else ""))
                    self.assertEqual(updated["fields"]["organic_description"]["quote"], inference)

    def test_explicit_unknown_geography_is_durable_without_erasing_known_country(self):
        for message in (
            "Место точно не знаю",
            "Место неизвестно",
            "Не знаю где сделано фото",
            "Местонахождение неизвестно",
        ):
            with self.subTest(message=message):
                memory = record_evidence({}, message)
                self.assertEqual(memory["fields"]["location"], {
                    "status": "unknown", "quote": message,
                })
                memory = record_evidence(memory, "Продолжим")
                self.assertEqual(memory["fields"]["location"]["status"], "unknown")
                country = record_evidence({}, "Фото из Канады")
                known_location = dict(country["fields"]["location"])
                country = record_evidence(country, message)
                self.assertEqual(country["fields"]["location"], known_location)
                self.assertEqual(country["fields"]["location_detail"], {
                    "status": "unknown", "quote": message,
                })
        memory = record_evidence({}, "Фото из Канады, точнее место не знаю")
        self.assertEqual(memory["fields"]["location"]["status"], "known")
        self.assertIn("Канады", memory["fields"]["location"]["quote"])

    def test_ten_actual_visitor_messages_keep_verified_quotes_and_question_budget(self):
        for fixture in self.fixtures:
            with self.subTest(case=fixture["case_id"]):
                case = self.replay_second_message(fixture["case_id"])
                for fact in case.get("reported_features", []):
                    self.assertIn(fact["quote"], " ".join(fixture["question"].split()))
                for _ in range(4):
                    text, case = reply(case, "Продолжим")
                    self.assertLessEqual(text.count("?"), 1)
                    case = record_case(case, "Не знаю")
                self.assertLessEqual(len(case["asked_fields"]), 3)
                self.assertNotIn("pending_photo", case)

    def test_podzolic_colors_from_free_speech_reach_the_same_evidence_as_confirmation(
        self,
    ):
        case = self.replay_second_message(1)
        evidence = evaluate(case)["evidence"]["present"]
        self.assertIn("pale_horizon", evidence)
        self.assertIn("rust_lower", evidence)
        self.assertEqual(case["fields"]["texture"]["status"], "unknown")
        self.assertNotEqual(choose_question(case), "texture")
        self.assertEqual(evaluate(case)["candidates"], [])
        self.assertNotEqual(choose_question(case), "soil_structure")
        text, _ = reply(case, "Подведите итог")
        self.assertIn("Рыжевато-бурый материал ниже тоже учтён", text)
        self.assertIn("состав светлого материала", text)

    def test_known_country_survives_unknown_town(self):
        case = self.replay_second_message(2)
        self.assertEqual(case["fields"]["location"]["status"], "known")
        self.assertIn("Канады", case["fields"]["location"]["quote"])
        self.assertEqual(case["fields"]["location_detail"]["status"], "unknown")
        self.assertEqual(choose_question(case), "organic_layer")

    def test_country_paraphrases_are_not_soil_or_geocoding_rules(self):
        for sentence in [
            "Город не знаю, но фото из Франции",
            "Это Бразилия, точный город не знаю",
        ]:
            case = record_case({}, sentence, updates=[])
            self.assertEqual(case["fields"]["location"]["status"], "known")
            self.assertEqual(evaluate(case)["candidates"], [])
        for sentence in [
            "Наверное, это Канада",
            "Это не Канада",
            "Не в Канаде",
            "Это Канада?",
            "Из Канады ли, я не знаю",
        ]:
            case = record_case({}, sentence, updates=[])
            self.assertNotIn("location", case["fields"], sentence)

    def test_both_unknown_water_and_vegetation_are_preserved_outside_model_cap(self):
        case = self.replay_second_message(8)
        self.assertEqual(case["fields"]["water"]["status"], "unknown")
        self.assertEqual(case["fields"]["landscape"]["status"], "unknown")
        self.assertNotIn("texture", case["fields"])
        self.assertNotIn(
            choose_question(case), {"landscape", "water", "soil_structure"}
        )

    def test_visible_water_is_distinct_from_ordinary_water_regime(self):
        case = self.replay_second_message(6)
        self.assertIn("surface_water", case["fields"])
        self.assertNotIn("water", case["fields"])
        self.assertIn("water_visible", evaluate(case)["evidence"]["present"])
        self.assertNotIn("long_waterlogging", evaluate(case)["evidence"]["present"])
        self.assertEqual(choose_question(case), "water")
        case = record_case(
            case,
            "После дождя вода быстро уходит",
            updates=[
                {
                    "field": "water",
                    "status": "known",
                    "quote": "После дождя вода быстро уходит",
                }
            ],
        )
        self.assertIn("water", case["fields"])

    def test_clods_and_stones_do_not_complete_texture(self):
        case = self.replay_second_message(3)
        self.assertNotIn("texture", case["fields"])
        self.assertIn("soil_structure", case["fields"])
        self.assertNotEqual(choose_question(case), "carbonate")
        old = known(texture="Камни", water="В выемке видна вода")
        updated = record_case(old, "Продолжим")
        self.assertIn("stones", updated["fields"])
        self.assertIn("surface_water", updated["fields"])
        self.assertNotIn("texture", updated["fields"])
        self.assertNotIn("water", updated["fields"])

    def test_absent_laboratory_results_never_trigger_specific_lab_questions(self):
        for message in ["Анализов нет", "Нет никаких анализов", "Анализов у меня нет"]:
            case = record_case(
                known(
                    location="Свердловская область",
                    organic_layer="Тёмный гумусовый горизонт",
                    soil_structure="Зернистая структура",
                ),
                message,
            )
            for _ in range(3):
                field = choose_question(case)
                self.assertNotIn(field, {"carbonate", "salinity", "sodicity"})
                _, case = reply(case, "Продолжим")
                case = record_case(case, "Не знаю")
        self.assertNotEqual(choose_question(self.replay_second_message(2)), "carbonate")

    def test_unresolved_position_can_be_clarified_despite_known_color(self):
        case = self.replay_second_message(7)
        self.assertIn("light_band", evaluate(case)["evidence"]["present"])
        self.assertNotIn("pale_horizon", evaluate(case)["evidence"]["present"])
        self.assertEqual(choose_question(case), "white_layer")
        photo = validated_observation(
            {
                "photo_id": "a" * 64,
                "scene": "profile",
                "quality": "usable",
                "color": "natural",
                "features": ["pale_horizon", "rust_lower"],
            }
        )
        case = record_case(
            case,
            "Ещё фото",
            new_photo=True,
            photo_description="Срез",
            photo_observation=photo,
        )
        text, case = reply(case, "Продолжим")
        self.assertEqual(case["pending_photo"]["feature"], "pale_horizon")
        self.assertEqual(text.count("?"), 1)

    def test_same_two_colors_on_a_saline_profile_do_not_select_podzols(self):
        for case in [
            known(profile="Внутри профиля белый слой, ниже него рыжий слой"),
            known(profile="Под светлым слоем рыжий песок внутри разреза"),
        ]:
            self.assertEqual(evaluate(case)["candidates"], [])
            text = render(evaluate(case))
            self.assertIn("состав", text)
            self.assertIn("солей", text)
            self.assertNotIn("\nПодзолы\n", text)

    def test_required_diagnostic_evidence_still_allows_a_source_linked_hypothesis(self):
        case = known(
            profile="Под светлым слоем рыжий слой внутри профиля. Почвовед описал альфегумусовый горизонт"
        )
        result = evaluate(case)
        self.assertIn("podzols", {row["card"]["id"] for row in result["candidates"]})
        self.assertIn("альфегумусовый", render(result))
        simple = known(
            profile="Внутри профиля светлый слой, ниже бурый",
            texture="Песчаный материал",
        )
        self.assertNotIn(
            "podzols", {row["card"]["id"] for row in evaluate(simple)["candidates"]}
        )

    def test_roots_on_orange_material_do_not_imply_a_dark_organic_layer(self):
        case = self.replay_second_message(10)
        self.assertIn("vegetation", case["fields"])
        self.assertNotEqual(choose_question(case), "organic_layer")

    def test_questions_hypotheses_and_uncertainty_never_become_reported_features(self):
        for message in [
            "Есть ли внутри белый слой, а ниже рыжий?",
            "Возможно, внутри белый слой, ниже бурый",
            "Если внутри будет светлый слой, это подзол?",
        ]:
            case = record_case({}, message, updates=[])
            self.assertEqual(case.get("reported_features", []), [])

    def test_model_cannot_cut_a_question_into_a_reported_diagnostic_claim(self):
        question = "Специалист описал альфегумусовый горизонт?"
        updates = validated_updates(
            question,
            [
                {
                    "field": "profile",
                    "status": "known",
                    "quote": "Специалист описал альфегумусовый горизонт",
                }
            ],
        )
        self.assertEqual(updates, {})
        statement = "Внутри профиля светлый слой. Ниже бурый."
        updates = validated_updates(
            statement, [{"field": "profile", "status": "known", "quote": statement}]
        )
        self.assertEqual(updates["profile"]["quote"], statement)

    def test_reported_regime_and_material_are_not_asked_again_under_different_fields(
        self,
    ):
        case = record_case(
            known(location="Свердловская область"),
            "После дождя вода быстро уходит. Под светлым слоем внутри разреза рыжий песок.",
            updates=[
                {
                    "field": "lower_layer",
                    "status": "known",
                    "quote": "Под светлым слоем внутри разреза рыжий песок",
                }
            ],
        )
        self.assertEqual(case["fields"]["water"]["status"], "known")
        self.assertNotIn(
            choose_question(case), {"water", "texture", "lower_layer", "white_layer"}
        )

    def test_existing_usable_profile_and_confirmed_layer_are_accounted_for_in_final_advice(
        self,
    ):
        photo = validated_observation(
            {
                "photo_id": "a" * 64,
                "scene": "profile",
                "quality": "usable",
                "color": "natural",
                "features": ["pale_horizon", "rust_lower"],
            }
        )
        case = record_case(
            known(location="Свердловская область"),
            "Фото",
            new_photo=True,
            photo_description="Срез",
            photo_observation=photo,
        )
        _, case = reply(case, "Продолжим")
        case = record_case(case, "Да")
        _, case = reply(case, "Да")
        case = record_case(case, "Да")
        text, _ = reply(case, "Подведите итог")
        self.assertIn("Положение уже описанного или подтверждённого слоя учтено", text)
        self.assertNotIn("уточнить, где именно расположен", text)
        self.assertNotIn("описание горизонтов уже доступного профиля с масштабом", text)
        for malformed in [
            None,
            [None],
            [
                {
                    "feature": "pale_horizon",
                    "status": {},
                    "quote": "Светлый слой",
                    "source": "user_report",
                }
            ],
        ]:
            case = record_case({"reported_features": malformed}, "Продолжим")
            self.assertEqual(case.get("reported_features", []), [])

    def test_reported_observations_survive_note_eviction_and_reset_with_site(self):
        case = record_case({}, "Внутри профиля светлый слой, ниже бурый")
        for _ in range(12):
            case = record_case(case, "Продолжим")
        self.assertIn("pale_horizon", evaluate(case)["evidence"]["present"])
        self.assertLessEqual(len(case["reported_features"]), 32)
        case = record_case(case, "Новый участок", reset=True)
        self.assertNotIn("reported_features", case)

    def test_malformed_legacy_field_rows_do_not_break_normalization(self):
        case = record_case(
            {
                "fields": {
                    "texture": None,
                    "water": [],
                    "location": {"status": {}, "quote": "text"},
                }
            },
            "Продолжим",
        )
        self.assertEqual(case["fields"], {})
        self.assertEqual(
            evaluate({"fields": {"texture": None, "water": []}})["candidates"], []
        )


if __name__ == "__main__":
    unittest.main()
