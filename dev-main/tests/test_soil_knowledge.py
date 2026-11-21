"""Behavioural regression for hypotheses; not an accuracy/soil-expert benchmark."""

import unittest
from soil_knowledge import evaluate, render, reference
from soil_interview import reply, choose_question


def case(**quotes):
    return {
        "fields": {
            key: {"status": "known", "quote": text} for key, text in quotes.items()
        }
    }


def ids(result):
    return {x["card"]["id"] for x in result["candidates"]}


class SoilKnowledgeTests(unittest.TestCase):
    def test_region_color_and_photo_are_not_a_diagnosis(self):
        c = case(location="Ленинградская область", organic_layer="Очень тёмная земля")
        self.assertEqual(ids(evaluate(c)), set())
        text, _ = reply(c, "Подведите итог", "Белый слой, ниже рыжий, песок. O-E-BHF-C")
        self.assertNotIn("Источник:", text)
        self.assertNotIn("Подзолы", text)

    def test_two_colors_give_explanation_instead_of_unsupported_podzol(self):
        result = evaluate(
            case(profile="Под светлым слоем лежит рыжий слой внутри профиля")
        )
        self.assertEqual(ids(result), set())
        text = render(result)
        self.assertIn("состав", text)
        self.assertIn("https://soils.narod.ru/taxon/type/podz07.html", text)
        self.assertIn("не различает процессы", text)
        self.assertNotIn("\nПодзолы\n", text)

    def test_russian_podzols_are_not_podzolic_or_gray_by_name(self):
        self.assertEqual(
            ids(evaluate(case(profile="Специалист записал O-E-BHF-C"))), {"podzols"}
        )
        self.assertEqual(
            ids(evaluate(case(profile="Специалист записал O-EL-BEL-BT-C"))),
            {"podzolic"},
        )
        self.assertEqual(
            ids(evaluate(case(profile="Специалист записал AY-AEL-BEL-BT-C"))), {"gray"}
        )

    def test_unknown_or_guessed_formula_is_not_evidence(self):
        c = {
            "fields": {"profile": {"status": "unknown", "quote": "Не знаю, O-E-BHF-C"}}
        }
        self.assertEqual(ids(evaluate(c)), set())
        self.assertEqual(ids(evaluate(case(profile="Возможно, O-E-BHF-C"))), set())
        self.assertEqual(
            ids(evaluate({"user_notes": ["O-E-BHF-C"], "photos": ["O-E-BHF-C"]})), set()
        )

    def test_white_crust_and_ph_are_not_soluble_salts_or_sodium(self):
        result = evaluate(
            case(white_layer="Белый налёт на поверхности", tests="Анализ: pH 8,5")
        )
        self.assertNotIn("solonchaks", ids(result))
        self.assertNotIn("solonetz", ids(result))

    def test_salt_and_exchangeable_sodium_are_separate(self):
        salt = case(
            white_layer="Белый налёт на поверхности",
            salinity="Лабораторный анализ: повышены водорастворимые соли",
        )
        self.assertEqual(ids(evaluate(salt)), {"solonchaks"})
        sodium = case(
            soil_structure="Столбчатая структура плотного горизонта",
            sodicity="Лабораторный анализ: повышенный обменный натрий",
        )
        self.assertEqual(ids(evaluate(sodium)), {"solonetz"})
        total = case(
            soil_structure="Столбчатая структура",
            tests="Лабораторный анализ: высокий общий натрий в воде",
        )
        self.assertNotIn("solonetz", ids(evaluate(total)))

    def test_imported_ground_suspends_natural_type_hypotheses(self):
        c = case(profile="O-E-BHF-C", origin="Завозили покупной грунт и торф")
        result = evaluate(c)
        self.assertEqual(ids(result), set())
        self.assertTrue(result["evidence"]["imported"])
        self.assertIn("привозном", render(result))
        self.assertIsNone(choose_question(c))

    def test_negation_and_contradiction_do_not_silently_choose_type(self):
        negative = case(profile="В профиле нет светлого слоя под подстилкой")
        result = evaluate(negative)
        self.assertIn("pale_horizon", result["evidence"]["absent"])
        conflict = case(
            profile="Внутри профиля светлый слой, ниже него бурый",
            white_layer="Внутри профиля нет светлого слоя",
        )
        result = evaluate(conflict)
        self.assertIn("pale_horizon", result["evidence"]["conflicts"])
        self.assertEqual(ids(result), set())
        self.assertIn("расходятся сведения", render(result))

    def test_sand_and_simple_o_c_formula_are_not_automatically_psammozem(self):
        self.assertEqual(ids(evaluate(case(texture="Песчаная почва"))), set())
        self.assertNotIn("psammozems", ids(evaluate(case(profile="O-C"))))
        c = case(
            texture="Песчаный материал",
            profile="Подстилка, сразу песок; развитых горизонтов нет",
        )
        self.assertIn("psammozems", ids(evaluate(c)))
        self.assertEqual(ids(evaluate(case(profile="WRB Arenosol"))), set())

    def test_dark_color_is_not_peat_or_chernozem(self):
        self.assertEqual(
            ids(
                evaluate(case(organic_layer="Чёрная почва", water="Вода стоит неделю"))
            ),
            set(),
        )
        peat = case(
            organic_layer="Толща из растительных остатков и волокон торфа",
            water="Вода стоит неделями",
        )
        result = evaluate(peat)
        self.assertNotIn("peat", ids(result))
        peat["fields"]["organic_depth"] = {
            "status": "known",
            "quote": "Толща торфа 70 см",
        }
        self.assertIn("peat", ids(evaluate(peat)))

    def test_roots_and_scattered_plant_fragments_are_not_a_peat_thickness(self):
        for description in [
            "Гумусовый слой с волокнами травы",
            "Слой земли с отдельными растительными остатками и корнями",
        ]:
            result = evaluate(
                case(organic_layer=description, water="Вода стоит неделями")
            )
            self.assertNotIn("peat", ids(result))

    def test_imported_material_is_detected_in_profile_not_only_origin_field(self):
        result = evaluate(case(profile="Завезли покупной торф; O-E-BHF-C"))
        self.assertTrue(result["evidence"]["imported"])
        self.assertEqual(ids(result), set())

    def test_formal_profile_must_match_whole_sequence(self):
        for formula in ["O-E-BHF-C-X", "X-O-E-BHF-C", "O-E-BHF-CG"]:
            self.assertNotIn("podzols", ids(evaluate(case(profile=formula))))

    def test_no_humus_values_are_invented_from_granular_black_soil(self):
        c = case(
            organic_layer="Тёмный гумусовый горизонт",
            soil_structure="Зернистая структура верхнего слоя",
        )
        result = evaluate(c)
        self.assertNotIn("chernozems", ids(result))
        c["fields"]["carbonate"] = {
            "status": "known",
            "quote": "Лабораторный анализ: карбонаты в нижнем горизонте",
        }
        result = evaluate(c)
        self.assertIn("chernozems", ids(result))
        text = render(result)
        self.assertIn("карбонат", text)
        self.assertNotIn("гумус 8%", text)
        self.assertNotIn("вероятность 90", text)

    def test_next_question_distinguishes_profile_instead_of_universal_ball_test(self):
        c = case(
            location="Свердловская область",
            profile="Внутри профиля под подстилкой светлый слой",
        )
        self.assertEqual(choose_question(c), "lower_layer")
        c["fields"]["lower_layer"] = {"status": "unknown", "quote": "Не знаю"}
        c["asked_fields"] = ["location", "lower_layer"]
        self.assertNotEqual(choose_question(c), "lower_layer")
        c["asked_fields"].append("clay_change")
        self.assertIsNone(choose_question(c))

    def test_question_does_not_presuppose_an_unreported_white_layer(self):
        c = case(
            location="Свердловская область", organic_layer="Серый гумусовый горизонт"
        )
        self.assertEqual(choose_question(c), "clay_change")

    def test_plain_language_structure_reply_can_be_saved_without_classifier_help(self):
        from soil_case import record_case

        c = case(sodicity="Лабораторный анализ: повышенный обменный натрий")
        c["pending_field"] = "soil_structure"
        c["asked_fields"] = ["soil_structure"]
        updated = record_case(c, "Внизу вертикальные столбики")
        self.assertEqual(
            updated["fields"]["soil_structure"]["quote"], "Внизу вертикальные столбики"
        )
        self.assertIn("solonetz", ids(evaluate(updated)))

    def test_short_reply_is_bound_to_the_programs_previous_question(self):
        c = case(
            profile="В профиле под подстилкой светлый слой", lower_layer="Рыжий песок"
        )
        self.assertNotIn("rust_lower", evaluate(c)["evidence"]["present"])
        c["asked_fields"] = ["lower_layer"]
        self.assertIn("rust_lower", evaluate(c)["evidence"]["present"])

    def test_peat_depth_reply_needs_the_matching_measurement_question(self):
        from soil_case import record_case

        c = case(
            organic_layer="Толща из растительных остатков и торфа",
            organic_depth="60 см",
        )
        self.assertNotIn("peat_thickness", evaluate(c)["evidence"]["present"])
        c["asked_fields"] = ["organic_depth"]
        c["pending_field"] = "organic_depth"
        updated = record_case(c, "65 см")
        self.assertEqual(updated["fields"]["organic_depth"]["quote"], "65 см")
        self.assertIn("peat_thickness", evaluate(updated)["evidence"]["present"])

    def test_missing_lab_analysis_is_unknown_not_a_measurement(self):
        from soil_case import record_case

        for text in ["Анализа нет", "Нет анализа", "Не делал анализ"]:
            c = record_case(
                {"pending_field": "salinity"},
                text,
                updates=[{"field": "salinity", "status": "unknown", "quote": text}],
            )
            self.assertEqual(c["fields"]["salinity"]["status"], "unknown")
            self.assertNotIn("soluble_salts", evaluate(c)["evidence"]["present"])

    def test_sources_and_review_status_are_explicit(self):
        data = reference()
        self.assertEqual(len(data["cards"]), 10)
        self.assertEqual(len({x["id"] for x in data["cards"]}), 10)
        for card in data["cards"]:
            self.assertEqual(
                card["review_status"], "source_checked_not_expert_validated"
            )
            self.assertTrue(card["confirmation_needed"])
            for source in card["sources"]:
                self.assertTrue(
                    source["url"].startswith("https://soils.narod.ru/taxon/type/")
                )
                self.assertRegex(source["download_sha256"], r"^[a-f0-9]{64}$")
