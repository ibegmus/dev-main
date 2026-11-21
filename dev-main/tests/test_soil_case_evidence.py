"""Production soil-case evidence persists independently of the display history."""

import copy
import unittest

from soil_case import record_case
from soil_evidence import context


class SoilCaseEvidenceTests(unittest.TestCase):
    def test_photo_access_and_explicit_unknowns_survive_note_eviction(self):
        messages = (
            "У меня только фото",
            "Водный режим не знаю",
            "На ощупь не проверял",
            "Анализов нет",
        )
        case = {}
        for message in messages:
            case = record_case(case, message)
        for index in range(12):
            case = record_case(case, f"Дополнительное сообщение {index}")
        self.assertEqual(len(case["user_notes"]), 8)
        self.assertTrue(all(message not in case["user_notes"] for message in messages))
        evidence = case["evidence"]
        self.assertEqual(evidence["access"], {"status": "photo_only", "quote": messages[0]})
        for key, message in zip(("water", "texture", "tests"), messages[1:]):
            self.assertEqual(evidence["fields"][key], {"status": "unknown", "quote": message})
        self.assertIn("только фото / нет доступа к участку", context(evidence))

    def test_explicit_access_and_water_corrections_replace_prior_reports(self):
        case = record_case({}, "У меня только фото. Водный режим не знаю")
        corrected = record_case(case, "Теперь я на участке. Обычно вода быстро уходит")
        self.assertEqual(case["evidence"]["access"]["status"], "photo_only")
        self.assertEqual(corrected["evidence"]["access"]["status"], "on_site")
        self.assertEqual(corrected["evidence"]["fields"]["water"], {
            "status": "known", "quote": "Обычно вода быстро уходит",
        })
        corrected = record_case(corrected, "Сейчас я не на участке")
        self.assertEqual(corrected["evidence"]["access"]["status"], "unknown")

    def test_new_site_clears_durable_evidence_before_recording_its_message(self):
        case = record_case({}, "У меня только фото. Водный режим не знаю")
        case = record_case(case, "Новый участок. Теперь я на участке", reset=True)
        self.assertEqual(case["evidence"]["access"]["status"], "on_site")
        self.assertNotIn("water", case["evidence"]["fields"])
        self.assertNotIn("только фото", context(case["evidence"]))

    def test_legacy_migration_replays_only_visitor_notes_and_only_once(self):
        legacy = {
            "user_notes": ["У меня только фото", "Водный режим не знаю"],
            "fields": {"texture": {"status": "known", "quote": "На ощупь тяжёлая глина"}},
            "photos": ["Вода стоит постоянно. Много гумуса"],
            "history": [{"question": "Теперь я на участке", "answer": "Вода постоянно стоит"}],
            "assistant_answer": "На ощупь тяжёлая глина",
        }
        before = copy.deepcopy(legacy)
        case = record_case(legacy, "Продолжим")
        self.assertEqual(legacy, before)
        self.assertEqual(case["evidence"]["access"]["status"], "photo_only")
        self.assertEqual(case["evidence"]["fields"]["water"]["status"], "unknown")
        self.assertNotIn("texture", case["evidence"]["fields"])
        self.assertNotIn("organic_layer", case["evidence"]["fields"])
        case["user_notes"] = ["Теперь я на участке", "Вода стоит постоянно"]
        updated = record_case(case, "Продолжим")
        self.assertEqual(updated["evidence"], case["evidence"])

    def test_stale_questionnaire_pending_does_not_bind_a_generic_unknown(self):
        for previous in ({"pending_field": "water", "active": True},
                         {"pending_field": "water", "user_notes": ["Продолжим"]}):
            with self.subTest(previous=previous):
                case = record_case(previous, "Не знаю", updates=[{
                    "field": "water", "status": "unknown", "quote": "Не знаю",
                }])
                self.assertNotIn("water", case["evidence"]["fields"])
                self.assertNotIn("pending_field", case["evidence"])

    def test_controlled_nested_pending_binds_unknown_and_preserves_asked_fields(self):
        case = record_case({}, "Определи почву")
        case["evidence"]["pending_field"] = "water"
        case["evidence"]["asked_fields"] = ["location", "water"]
        case["pending_field"] = "texture"
        case = record_case(case, "Не знаю")
        self.assertEqual(case["evidence"]["fields"]["water"], {
            "status": "unknown", "quote": "Не знаю", "question":
                "После дождя вода здесь быстро уходит или долго стоит на поверхности?",
        })
        self.assertNotIn("texture", case["evidence"]["fields"])
        self.assertNotIn("pending_field", case["evidence"])
        self.assertEqual(case["evidence"]["asked_fields"], ["location", "water"])
        for index in range(12):
            case = record_case(case, f"Дополнительное сообщение {index}")
        self.assertEqual(case["evidence"]["fields"]["water"]["status"], "unknown")
        self.assertEqual(case["evidence"]["asked_fields"], ["location", "water"])

    def test_latest_updates_require_a_visitor_quote_and_preserve_unknown(self):
        case = record_case({}, "Место неизвестно", updates=[
            {"field": "location", "status": "unknown", "quote": "Место неизвестно"},
            {"field": "water", "status": "known", "quote": "Вода постоянно стоит"},
        ])
        self.assertEqual(case["evidence"]["fields"], {
            "location": {"status": "unknown", "quote": "Место неизвестно"},
        })

    def test_access_answers_bind_only_to_the_controlled_nested_access_question(self):
        for status, answers in (
            ("on_site", ("Да", "Есть", "Могу")),
            ("photo_only", ("Нет", "Нет доступа")),
            ("unknown", ("Не знаю", "Не могу сейчас")),
        ):
            for answer in answers:
                with self.subTest(answer=answer):
                    isolated = record_case({}, answer)
                    self.assertEqual(isolated["evidence"]["access"]["status"], "unknown")
                    legacy = record_case({"pending_field": "access"}, answer)
                    self.assertEqual(legacy["evidence"]["access"]["status"], "unknown")
                    controlled = record_case({}, "Определи почву")
                    controlled["evidence"]["pending_field"] = "access"
                    controlled["evidence"]["asked_fields"] = ["access"]
                    controlled = record_case(controlled, answer)
                    access = controlled["evidence"]["access"]
                    self.assertEqual(access["status"], status)
                    self.assertEqual(access["quote"], answer)
                    self.assertEqual(access["question_field"], "access")
                    self.assertNotIn("pending_field", controlled["evidence"])
                    self.assertEqual(controlled["evidence"]["asked_fields"], ["access"])
                    later = record_case(controlled, "Продолжим")
                    self.assertEqual(later["evidence"]["access"], access)

    def test_explicit_sample_access_does_not_require_being_on_the_plot(self):
        for message in ("Образец под рукой", "Держу землю в руках", "У меня есть образец"):
            with self.subTest(message=message):
                case = record_case({}, message)
                self.assertEqual(case["evidence"]["access"], {"status": "on_site", "quote": message})
        for message in ("Образец не под рукой", "Не держу землю в руках", "Если будет образец под рукой"):
            with self.subTest(message=message):
                self.assertEqual(record_case({}, message)["evidence"]["access"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
