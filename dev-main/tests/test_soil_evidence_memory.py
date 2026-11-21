"""User evidence regressions; these tests make no model-accuracy claim."""

import copy
import unittest

from soil_evidence import context, record, controlled_access_answer


def field(key, quote, status="known"):
    return {key: {"status": status, "quote": quote}}


class SoilEvidenceMemoryTests(unittest.TestCase):
    def test_access_persists_independently_of_eight_note_window(self):
        first = "У меня  только фото."
        memory = record({}, first)
        self.assertEqual(memory["access"], {"status": "photo_only", "quote": first})
        for index in range(15):
            memory = record({"evidence": memory, "user_notes": ["другое"] * 8},
                            f"Дополнительное сообщение {index}")
        self.assertEqual(memory["access"]["status"], "photo_only")
        self.assertEqual(memory["access"]["quote"], first)
        self.assertIn(first, context(memory))

    def test_explicit_access_corrections_replace_prior_access(self):
        memory = record({}, "Не могу добраться до участка")
        self.assertEqual(memory["access"]["status"], "photo_only")
        memory = record(memory, "Теперь я на участке")
        self.assertEqual(memory["access"], {"status": "on_site", "quote": "Теперь я на участке"})
        memory = record(memory, "Сейчас я не на участке")
        self.assertEqual(memory["access"]["status"], "unknown")
        memory = record(memory, "Доступа к участку нет")
        self.assertEqual(memory["access"]["status"], "photo_only")

    def test_controlled_access_negative_equivalents_are_not_standalone_access_facts(self):
        for answer in ("пока нет", "сейчас нет"):
            with self.subTest(answer=answer):
                expected = {"status": "photo_only", "quote": answer, "question_field": "access"}
                self.assertEqual(controlled_access_answer(answer), expected)
                self.assertEqual(record({"pending_field": "access"}, answer)["access"], expected)
                self.assertEqual(record({}, answer)["access"]["status"], "unknown")
                later = record({"pending_field": "access"}, answer)
                later = record(later, "Теперь я на участке")
                self.assertEqual(later["access"]["status"], "on_site")

    def test_brief_origin_answers_bind_only_to_the_controlled_origin_question(self):
        for answer in ("местная", "местный", "привозная", "привозной"):
            with self.subTest(answer=answer):
                memory = record({"pending_field": "origin"}, answer)
                self.assertEqual(memory["fields"]["origin"]["status"], "known")
                self.assertEqual(memory["fields"]["origin"]["quote"], answer)
                self.assertIn("question", memory["fields"]["origin"])
                for supplied in (None, field("origin", answer)):
                    self.assertNotIn("origin", record({}, answer, supplied)["fields"])
                    self.assertNotIn("origin", record({"pending_field": "landscape"}, answer, supplied)["fields"])
        for answer in ("Не знаю", "Не уверен, местная", "Не знаю, местная или привозная"):
            with self.subTest(answer=answer):
                origin = record({"pending_field": "origin"}, answer)["fields"]["origin"]
                self.assertEqual(origin["status"], "unknown")
                self.assertEqual(origin["quote"], answer)
        uncertain = record({"pending_field": "origin"}, "Не уверен, местная", field("origin", "местная"))
        self.assertEqual(uncertain["fields"]["origin"], {
            "status": "unknown", "quote": "Не уверен, местная",
            "question": "Известно ли, завозили сюда грунт, торф или компост?",
        })
        for answer in ("Возможно местная", "Привозная?", "Не местная"):
            with self.subTest(answer=answer):
                self.assertNotEqual(record({"pending_field": "origin"}, answer)["fields"].get("origin", {}).get("status"), "known")

    def test_access_negation_questions_and_hypotheses_do_not_create_facts(self):
        for message in (
            "У меня не только фото", "Я не на участке", "Я на участке?",
            "Когда приеду на участок", "Если буду на участке",
            "Вы сказали: у меня только фото", "Неправда, что я на участке",
        ):
            with self.subTest(message=message):
                self.assertEqual(record({}, message)["access"]["status"], "unknown")
        memory = record({}, "У меня не только фото, я сейчас на участке")
        self.assertEqual(memory["access"]["status"], "on_site")
        memory = record({}, "Не знаю, смогу ли попасть на участок")
        self.assertEqual(memory["access"]["status"], "unknown")
        self.assertEqual(memory["access"]["quote"], "Не знаю, смогу ли попасть на участок")

    def test_live_demonstrative_photo_only_wording_and_its_negation(self):
        for message in (
            "У меня только эта фотография.",
            "Место неизвестно, у меня только это фото.",
        ):
            with self.subTest(message=message):
                memory = record({}, message)
                self.assertEqual(memory["access"]["status"], "photo_only")
                self.assertEqual(memory["access"]["quote"], message)
        for message in ("У меня не только это фото", "У меня не только эта фотография"):
            self.assertEqual(record({}, message)["access"]["status"], "unknown")

    def test_photo_and_assistant_history_never_seed_user_evidence(self):
        legacy = {
            "version": 4,
            "fields": field("water", "Вода постоянно стоит"),
            "photos": ["Чёрная тяжёлая постоянно мокрая почва"],
            "photo_description": "Вода постоянно стоит",
            "assistant_answer": "Это торф и тяжёлая глина",
            "user_notes": ["У меня только фото"],
        }
        memory = record(legacy, "Продолжим")
        self.assertEqual(memory["fields"], {})
        self.assertEqual(memory["access"]["status"], "unknown")
        self.assertNotIn("постоянно стоит", context(legacy))
        self.assertNotIn("тяжёлая", context(memory))

    def test_provided_rows_need_quotes_from_latest_visitor_message(self):
        memory = record({}, "На фото вода", {
            "water": {"status": "known", "quote": "Всегда стоит вода"},
            "texture": {"status": "known", "quote": "Тяжёлая глина"},
        })
        self.assertNotIn("water", memory["fields"])
        self.assertNotIn("texture", memory["fields"])
        self.assertIn("surface_water", memory["fields"])
        self.assertNotIn("Тяжёлая глина", context(memory))

    def test_surface_water_does_not_imply_regime_even_after_rain(self):
        message = "На фото вода после дождя"
        memory = record({}, message, field("water", message))
        self.assertNotIn("water", memory["fields"])
        self.assertEqual(memory["fields"]["surface_water"]["quote"], message)
        memory = record(memory, "Обычно вода после дождя быстро уходит")
        self.assertEqual(memory["fields"]["water"]["status"], "known")
        self.assertIn("быстро уходит", memory["fields"]["water"]["quote"])
        self.assertIn("surface_water", memory["fields"])

    def test_unknown_water_and_reported_visible_water_are_separate(self):
        memory = record({}, "Не знаю, всегда ли тут сыро; на фото вода")
        self.assertEqual(memory["fields"]["water"]["status"], "unknown")
        self.assertEqual(memory["fields"]["surface_water"]["status"], "known")
        text = context(memory)
        self.assertIn("Обычный водный режим: неизвестно со слов посетителя", text)
        self.assertIn("Наличие воды на снимке: сообщено посетителем", text)

    def test_direct_report_of_constant_water_is_retained_as_a_report(self):
        memory = record({}, "Вода стоит постоянно")
        self.assertEqual(memory["fields"]["water"], {
            "status": "known", "quote": "Вода стоит постоянно",
        })
        self.assertIn("Обычный водный режим: сообщено посетителем", context(memory))

    def test_clods_color_and_visual_clay_are_not_tactile_measurements(self):
        for message in ("В земле крупные комочки", "Тёмная земля", "Похожа на глину", "Липкая на вид"):
            with self.subTest(message=message):
                memory = record({}, message, field("texture", message))
                self.assertNotIn("texture", memory["fields"])
        memory = record({}, "В земле крупные комочки", field("texture", "В земле крупные комочки"))
        self.assertIn("soil_structure", memory["fields"])
        memory = record({}, "На ощупь липкая, скатал шарик и ленту")
        self.assertEqual(memory["fields"]["texture"]["status"], "known")

    def test_dark_color_does_not_confirm_organic_material(self):
        message = "Тёмная земля"
        memory = record({}, message, field("organic_layer", message))
        self.assertIn("surface_color", memory["fields"])
        self.assertNotIn("organic_layer", memory["fields"])
        memory = record(memory, "Не знаю, есть ли там торф")
        self.assertEqual(memory["fields"]["organic_layer"]["status"], "unknown")
        memory = record(memory, "Различимы растительные остатки")
        self.assertEqual(memory["fields"]["organic_layer"]["status"], "known")
        self.assertIn("не заменяют анализ", context(memory))

    def test_brief_answers_bind_only_to_existing_program_field(self):
        memory = record({"pending_field": "texture"}, "Рассыпается")
        self.assertEqual(memory["fields"]["texture"]["quote"], "Рассыпается")
        self.assertIn("question", memory["fields"]["texture"])
        memory = record(memory, "Продолжим")
        self.assertIn("texture", memory["fields"])
        self.assertIn("ручной проверке: сообщено", context(memory))
        memory = record({"pending_field": "water"}, "Быстро уходит")
        self.assertEqual(memory["fields"]["water"]["quote"], "Быстро уходит")
        memory = record({"pending_field": "texture"}, "Не пробовал")
        self.assertEqual(memory["fields"]["texture"]["status"], "unknown")
        for previous in ({}, {"pending_field": "invented"}, {"pending_field": "water"}):
            self.assertNotIn("texture", record(previous, "Да")["fields"])

    def test_compact_controlled_location_answer_keeps_its_quote_and_safe_expansion(self):
        for answer in ("в екб", "Екб", "ЕКБ", "В Екб", "  в  Екб  "):
            with self.subTest(answer=answer):
                memory = record({"pending_field": "location"}, answer)
                location = memory["fields"]["location"]
                self.assertEqual(location["status"], "known")
                self.assertEqual(location["quote"], answer.strip())
                self.assertEqual(location["normalized"], "Екатеринбург")
                self.assertIn("question", location)
                self.assertNotIn("область", location["normalized"])
                self.assertIn("Екатеринбург", context(memory))
                self.assertEqual(record(memory, "Продолжим")["fields"]["location"], location)

    def test_compact_location_is_not_bound_without_a_location_question_or_phrase(self):
        for answer in ("Екб", "ЕКБ", "Киров"):
            for supplied in (None, field("location", answer)):
                with self.subTest(answer=answer, supplied=supplied):
                    self.assertNotIn("location", record({}, answer, supplied)["fields"])
                    self.assertNotIn("location", record({"pending_field": "water"}, answer, supplied)["fields"])
        self.assertEqual(record({}, "в екб")["fields"]["location"]["normalized"], "Екатеринбург")
        for answer in ("Да", "Не знаю", "Не уверен, Екб", "Екб?", "Вы сказали: Екб"):
            with self.subTest(answer=answer):
                location = record({"pending_field": "location"}, answer)["fields"].get("location", {})
                self.assertNotEqual(location.get("status"), "known")
        uncertain = record({"pending_field": "location"}, "Не уверен, в Екб", field("location", "Екб"))
        self.assertNotEqual(uncertain["fields"].get("location", {}).get("status"), "known")

    def test_compact_town_answer_preserves_the_reported_region_and_never_geocodes_ambiguity(self):
        original = record({}, "Свердловская область")
        for answer in ("в екб", "Екб", "Киров"):
            with self.subTest(answer=answer):
                memory = record({"evidence": original, "pending_field": "location"}, answer)
                self.assertEqual(memory["fields"]["location"], original["fields"]["location"])
                detail = memory["fields"]["location_detail"]
                self.assertEqual(detail["quote"], answer)
                self.assertIn("question", detail)
                self.assertEqual(record(memory, "Продолжим")["fields"]["location_detail"], detail)
                if answer == "Киров":
                    self.assertNotIn("normalized", detail)
        arbitrary = record({"pending_field": "location"}, "Киров", {
            "location": {"status": "known", "quote": "Киров", "normalized": "Екатеринбург"},
        })
        self.assertNotIn("normalized", arbitrary["fields"]["location"])

    def test_generic_unknown_cannot_be_assigned_to_an_unasked_field(self):
        memory = record({"pending_field": "location"}, "Не знаю", field("water", "Не знаю", "unknown"))
        self.assertEqual(memory["fields"]["location"]["status"], "unknown")
        self.assertNotIn("water", memory["fields"])
        self.assertEqual(record({}, "Не знаю", field("water", "Не знаю", "unknown"))["fields"], {})

    def test_hypothetical_or_assistant_attributed_traits_do_not_become_known(self):
        for message in (
            "Если я буду на участке, вода может стоять неделями",
            "Вы сказали, что вода стоит постоянно",
            "Если вода долго стоит, это глей?",
            "Возможно, внутри растительные остатки",
            "Вы написали: Свердловская область",
        ):
            with self.subTest(message=message):
                memory = record({}, message, field("water", message))
                self.assertEqual(memory["fields"], {})
                self.assertEqual(memory["access"]["status"], "unknown")

    def test_negated_physical_reports_keep_the_negative_quote(self):
        memory = record({}, "На фото воды нет. Растительных остатков нет. Обычно вода не застаивается.")
        self.assertIn("воды нет", memory["fields"]["surface_water"]["quote"])
        self.assertIn("остатков нет", memory["fields"]["organic_layer"]["quote"])
        self.assertIn("не застаивается", memory["fields"]["water"]["quote"])
        self.assertNotIn("получен торф", context(memory))

    def test_known_location_and_explicit_unknowns_survive_later_messages(self):
        memory = record({}, "Свердловская область")
        memory = record({"evidence": memory, "pending_field": "texture"}, "Не проверял")
        memory = record({"evidence": memory, "pending_field": "water"}, "Не знаю")
        for _ in range(12):
            memory = record(memory, "Продолжим")
        self.assertEqual(memory["fields"]["location"]["quote"], "Свердловская область")
        self.assertEqual(memory["fields"]["texture"]["status"], "unknown")
        self.assertEqual(memory["fields"]["water"]["status"], "unknown")

    def test_explicit_region_retraction_replaces_the_old_known_region(self):
        original = record({}, "Свердловская область")
        for correction in (
            "Я ошибся, регион мне неизвестен",
            "Участок не в Свердловской области, место неизвестно",
        ):
            with self.subTest(correction=correction):
                memory = record(original, correction)
                self.assertEqual(memory["fields"]["location"], {
                    "status": "unknown", "quote": correction,
                })
                self.assertEqual(record(memory, "Продолжим")["fields"]["location"]["status"], "unknown")
        unchanged = record(original, "Не знаю", field("location", "Не знаю", "unknown"))
        self.assertEqual(unchanged["fields"]["location"], original["fields"]["location"])

    def test_uncertain_region_mentions_keep_the_complete_qualifier(self):
        for message, extracted in (
            ("Не уверен, Свердловская область", "Свердловская область"),
            ("Не знаю, Свердловская область это или Пермский край", "Пермский край"),
            ("Свердловская область, не уверен", "Свердловская область"),
        ):
            for supplied in (None, field("location", extracted)):
                with self.subTest(message=message, supplied=supplied):
                    memory = record({}, message, supplied)
                    self.assertEqual(memory["fields"]["location"], {
                        "status": "unknown", "quote": message,
                    })
        certain = record({}, "Свердловская область, город не знаю")
        self.assertEqual(certain["fields"]["location"], {
            "status": "known", "quote": "Свердловская область",
        })
        country = record({}, "Фото из Канады")
        memory = record(country, "Не уверен, Свердловская область")
        self.assertEqual(memory["fields"]["location"], country["fields"]["location"])
        self.assertEqual(memory["fields"]["location_detail"]["status"], "unknown")

    def test_unknown_city_or_region_does_not_retract_the_known_country(self):
        original = record({}, "Фото из Канады")
        for message in ("Точный город не знаю", "Я ошибся, регион мне неизвестен"):
            with self.subTest(message=message):
                memory = record(original, message, field("location", message, "unknown"))
                self.assertEqual(memory["fields"]["location"], original["fields"]["location"])
                self.assertEqual(memory["fields"]["location_detail"], {
                    "status": "unknown", "quote": message,
                })

    def test_unknown_town_does_not_erase_reported_country(self):
        memory = record({}, "Фото из Канады", field("location", "из Канады"))
        memory = record(memory, "Город не знаю", field("location", "Город не знаю", "unknown"))
        self.assertEqual(memory["fields"]["location"], {"status": "known", "quote": "из Канады"})
        self.assertEqual(memory["fields"]["location_detail"]["status"], "unknown")

    def test_explicit_unperformed_touch_test_is_unknown(self):
        for message in ("На ощупь не пробовал", "Руками не трогал"):
            with self.subTest(message=message):
                memory = record({}, message)
                self.assertEqual(memory["fields"]["texture"]["status"], "unknown")

    def test_exact_user_whitespace_is_retained_without_mutating_prior_memory(self):
        message = "На ощупь  держит\tшарик"
        memory = record({}, message, field("texture", "На ощупь держит шарик"))
        self.assertEqual(memory["fields"]["texture"]["quote"], message)
        before = copy.deepcopy(memory)
        record(memory, "Теперь я на участке")
        self.assertEqual(memory, before)

    def test_malformed_optional_state_does_not_break_recording_or_context(self):
        memory = record({"pending_field": []}, None, [None, {"field": [], "quote": "a"}])
        memory["fields"] = {"water": {"status": [], "quote": "плохо"}}
        memory["access"] = {"status": [], "quote": None}
        self.assertIsInstance(record(memory, "Продолжим"), dict)
        self.assertIn("Доступ: доступ неизвестен", context(memory))

    def test_context_is_advisory_and_lists_unreported_core_properties(self):
        text = context(record({}, "У меня только фото"))
        self.assertIn("Обычный водный режим: неизвестно", text)
        self.assertIn("Механический состав по ручной проверке: неизвестно", text)
        self.assertIn("не устанавливает постоянное переувлажнение", text)
        self.assertIn("Тёмный цвет не измеряет", text)
        self.assertIn("предварительных гипотез", text)


if __name__ == "__main__":
    unittest.main()
