"""The practical continuation is deterministic and never calls a model."""
import unittest

from soil_next_step import action_plan, practical_followup


def evidence(status):
    return {'kind': 'user_soil_evidence', 'version': 1,
            'access': {'status': status, 'quote': 'Сообщение посетителя'}, 'fields': {}}


class SoilNextStepTests(unittest.TestCase):
    def test_short_practical_followups_and_frustration_are_recognized(self):
        for text in ('и че', 'и чё?', 'И ЧО?!', 'ну и что', 'и что теперь?',
                     'что мне делать', 'мне-то что делать', 'мне то что делать блять',
                     'блядь, и чё мне делать теперь?', 'что с этим делать?',
                     'мне что с этим делать', 'а дальше?', 'следующий шаг'):
            with self.subTest(text=text):
                self.assertTrue(practical_followup(text))

    def test_specific_new_topics_and_identification_are_not_hijacked(self):
        for text in ('что делать с сорняками', 'ну и что делать с сорняками блять',
                     'что это', 'и что это?', 'что такое чернозём',
                     'как измерить pH', 'что делать, если pH низкий?',
                     'что мне делать после измерения pH', 'как сделать фото',
                     'и что посадить на глине', 'какое удобрение выбрать',
                     'он спросил «и чё»', 'мне что делать с яблоней?',
                     '', 'Спасибо', None, 12):
            with self.subTest(text=text):
                self.assertFalse(practical_followup(text))

    def test_no_access_gets_one_realistic_later_observation(self):
        text = action_plan(evidence('photo_only'))
        self.assertIn('Сейчас ничего проверять не нужно', text)
        self.assertIn('Когда будете на участке', text)
        self.assertIn('за 2 минуты', text)
        self.assertIn('уже открытого среза', text)
        self.assertIn('при дневном свете', text)
        self.assertIn('поверхность и слои ниже', text)
        self.assertIn('отдельные слои или материал выглядит однородным', text)
        self.assertIn('специально копать и что-либо покупать не надо', text)
        self.assertNotIn('?', text)

    def test_on_site_access_is_used_without_demanding_a_new_excavation(self):
        text = action_plan(evidence('on_site'))
        self.assertTrue(text.startswith('Если рядом уже есть открытый срез земли'))
        self.assertNotIn('Когда будете на участке', text)
        self.assertIn('специально копать и что-либо покупать не надо', text)

    def test_legacy_or_invalid_access_does_not_become_an_on_site_fact(self):
        for value in ({'access': {'status': 'on_site'}},
                      {'kind': 'user_soil_evidence', 'version': 2, 'access': {'status': 'on_site'}},
                      {'kind': 'user_soil_evidence', 'version': 1, 'access': 'on_site'},
                      None, [], evidence('unknown')):
            with self.subTest(value=value):
                self.assertTrue(action_plan(value).startswith('Когда будет возможность'))

    def test_enclosing_soil_case_uses_only_its_durable_evidence(self):
        direct = action_plan(evidence('photo_only'))
        case = {'access': {'status': 'on_site'}, 'evidence': evidence('photo_only')}
        self.assertEqual(action_plan(case), direct)

    def test_followup_switches_to_practical_chat_help_without_repeating_the_photo_plan(self):
        for status in ('photo_only', 'unknown', 'on_site'):
            with self.subTest(status=status):
                text = action_plan(evidence(status), followup=True)
                self.assertIn('только на основании этой версии', text)
                self.assertIn('что собираетесь выращивать или какую проблему уже заметили', text)
                self.assertNotIn('срез', text)
                self.assertNotIn('участок', text)
                self.assertNotIn('снимок', text)
                self.assertNotIn('?', text)

    def test_no_plan_diagnoses_a_type_or_properties_from_the_observation(self):
        for status in ('photo_only', 'on_site', 'unknown'):
            for followup in (False, True):
                with self.subTest(status=status, followup=followup):
                    text = action_plan(evidence(status), followup=followup)
                    self.assertLess(len(text.split()), 100)
                    for claim in ('значит, это', 'это подзол', 'это псаммозём',
                                  'много гумуса', 'плохой дренаж', 'высокая глинистость'):
                        self.assertNotIn(claim, text.lower())
                    for pressure in ('купите', 'выкопайте', 'добавьте известь', 'внесите удобрения'):
                        self.assertNotIn(pressure, text.lower())

    def test_swear_words_are_data_and_do_not_reach_the_public_plan(self):
        message = 'мне то что делать блять'
        self.assertTrue(practical_followup(message))
        text = action_plan(evidence('photo_only'), followup=True)
        for phrase in ('блять', 'ругаться', 'вежлив', 'успокойтесь'):
            self.assertNotIn(phrase, text.lower())


if __name__ == '__main__':
    unittest.main()
