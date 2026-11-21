import copy
import unittest

from soil_evidence import record
from soil_followup import ALLOWED_CODES, FOLLOWUPS, choose_followup, resolve_followup


def evidence(*, access='on_site', fields=None, **markers):
    return {'kind': 'user_soil_evidence', 'version': 1,
            'access': {'status': access, 'quote': 'Сейчас я на участке.'},
            'fields': fields or {}, **markers}


class SoilFollowupTests(unittest.TestCase):
    def test_only_controlled_codes_can_produce_a_single_question(self):
        for selection in ('none', 'pH', 'Копайте яму глубиной метр?', '', None, [], {}, True):
            with self.subTest(selection=selection):
                self.assertEqual(choose_followup(selection, evidence()), '')
        self.assertEqual(set(ALLOWED_CODES), set(FOLLOWUPS) | {'none'})
        for code, followup in FOLLOWUPS.items():
            with self.subTest(code=code):
                text = choose_followup(code, evidence(access='unknown' if code == 'access' else 'on_site'))
                self.assertEqual(text, followup.text)
                self.assertEqual(text.count('?'), 1)
                self.assertLessEqual(followup.minutes, 10)
                self.assertLessEqual(followup.cost_rub, 100)

    def test_known_and_explicit_unknown_fields_are_not_asked_again_including_aliases(self):
        for code, followup in FOLLOWUPS.items():
            for field in followup.fields:
                for status in ('known', 'unknown'):
                    with self.subTest(code=code, field=field, status=status):
                        memory = evidence(fields={field: {'status': status, 'quote': 'Не знаю'}})
                        self.assertEqual(choose_followup(code, memory), '')

    def test_photo_only_blocks_all_physical_requests_but_allows_unanswered_known_information(self):
        memory = evidence(access='photo_only')
        for code in ('material', 'texture', 'profile'):
            self.assertEqual(choose_followup(code, memory), '')
        for code in ('location', 'landscape', 'imported', 'water_regime'):
            self.assertEqual(choose_followup(code, memory), FOLLOWUPS[code].text)

    def test_unknown_access_chooses_unanswered_information_before_a_physical_test(self):
        memory = evidence(access='unknown')
        self.assertEqual(choose_followup('texture', memory), FOLLOWUPS['location'].text)
        memory['fields']['location'] = {'status': 'known', 'quote': 'Тверская область'}
        self.assertEqual(choose_followup('texture', memory), FOLLOWUPS['landscape'].text)
        memory['fields']['landscape'] = {'status': 'unknown', 'quote': 'Не знаю'}
        self.assertEqual(choose_followup('material', memory), FOLLOWUPS['imported'].text)
        memory['fields']['origin'] = {'status': 'unknown', 'quote': 'Не знаю'}
        self.assertEqual(choose_followup('profile', memory), FOLLOWUPS['access'].text)
        memory['asked_fields'] = ['access']
        self.assertEqual(choose_followup('profile', memory), '')
        self.assertEqual(choose_followup('none', evidence(access='unknown')), '')

    def test_asked_and_pending_markers_prevent_repetition_without_mutation(self):
        for marker, value in (('asked_fields', ['water']), ('asked_fields', ['water_regime']),
                              ('pending_field', 'water')):
            case = {'evidence': evidence(), marker: value}
            previous = copy.deepcopy(case)
            self.assertEqual(choose_followup('water_regime', case), '')
            self.assertEqual(case, previous)
        self.assertEqual(choose_followup('texture', evidence(asked_fields=['texture'])), '')
        self.assertEqual(choose_followup('texture', evidence(access='unknown', asked_fields=['location'])),
                         FOLLOWUPS['landscape'].text)

    def test_resolver_identifies_actual_fallback_for_durable_pending_field(self):
        memory = evidence(access='unknown')
        for expected, field in (('location', 'location'), ('landscape', 'landscape'),
                                ('imported', 'origin'), ('access', 'access')):
            with self.subTest(expected=expected):
                resolved = resolve_followup('texture', memory)
                self.assertEqual(resolved, {'text': FOLLOWUPS[expected].text,
                                            'code': expected, 'field': field})
                memory['fields'][field] = {'status': 'unknown', 'quote': 'Не знаю'}
        self.assertIsNone(resolve_followup('texture', memory))
        self.assertIsNone(resolve_followup('texture', evidence(access='photo_only')))
        self.assertIsNone(resolve_followup('none', evidence()))

    def test_access_question_uses_durable_access_status_and_is_never_repeated(self):
        expected = {'text': 'У вас сейчас есть доступ к участку или образец земли под рукой?',
                    'code': 'access', 'field': 'access'}
        self.assertEqual(resolve_followup('access', evidence(access='unknown')), expected)
        for access in ('on_site', 'photo_only'):
            with self.subTest(access=access):
                self.assertIsNone(resolve_followup('access', evidence(access=access)))
        for markers in ({'asked_fields': ['access']}, {'pending_field': 'access'}):
            with self.subTest(markers=markers):
                self.assertIsNone(resolve_followup('access', evidence(access='unknown', **markers)))
                self.assertIsNone(resolve_followup('access', {
                    'evidence': evidence(access='unknown'), **markers}))

    def test_landscape_question_matches_remembered_vegetation_and_use(self):
        self.assertEqual(choose_followup('landscape', evidence()),
                         'Если знаете, что окружает место: лес, луг, поле или сад?')
        memory = record({'evidence': evidence(), 'pending_field': 'landscape'}, 'Лес.')
        self.assertEqual(memory['fields']['landscape']['status'], 'known')
        self.assertIsNone(resolve_followup('landscape', memory))

    def test_untried_simple_experiment_is_allowed_once_with_explicit_on_site_access(self):
        for code in ('material', 'texture'):
            field = FOLLOWUPS[code].fields[0]
            for quote in ('На ощупь не проверял.', 'Ещё не пробовала.', 'Землю не трогали.'):
                with self.subTest(code=code, quote=quote):
                    memory = evidence(fields={field: {'status': 'unknown', 'quote': quote}})
                    previous = copy.deepcopy(memory)
                    self.assertEqual(resolve_followup(code, memory),
                                     {'text': FOLLOWUPS[code].text, 'code': code, 'field': field})
                    self.assertEqual(memory, previous)
                    memory['asked_fields'] = [field]
                    self.assertIsNone(resolve_followup(code, memory))
                    del memory['asked_fields']
                    memory['pending_field'] = field
                    self.assertIsNone(resolve_followup(code, memory))

    def test_unknown_is_not_an_invitation_to_retry_refused_or_unavailable_experiments(self):
        for code in ('material', 'texture', 'profile'):
            field = FOLLOWUPS[code].fields[0]
            for quote in ('Не знаю.', 'Не проверял и не могу проверить.', 'Не могу, не пробовал.'):
                with self.subTest(code=code, quote=quote):
                    memory = evidence(fields={field: {'status': 'unknown', 'quote': quote}})
                    self.assertIsNone(resolve_followup(code, memory))
        for access in ('unknown', 'photo_only'):
            for code in ('material', 'texture'):
                field = FOLLOWUPS[code].fields[0]
                memory = evidence(access=access,
                                  fields={field: {'status': 'unknown', 'quote': 'Не проверял.'}})
                self.assertIsNone(resolve_followup(code, memory))
        self.assertIsNone(resolve_followup('profile', evidence(
            fields={'profile': {'status': 'unknown', 'quote': 'Не проверял.'}})))
        self.assertIsNone(resolve_followup('texture', evidence(
            fields={'texture': {'status': 'known', 'quote': 'Не проверял.'}})))

    def test_real_durable_untried_texture_can_offer_first_experiment_and_then_stop(self):
        memory = record({}, 'Сейчас я на участке. На ощупь не проверял.')
        self.assertEqual(memory['access']['status'], 'on_site')
        self.assertEqual(memory['fields']['texture']['status'], 'unknown')
        resolved = resolve_followup('texture', memory)
        self.assertEqual(resolved['code'], 'texture')
        self.assertEqual(resolved['field'], 'texture')
        memory['asked_fields'] = [resolved['field']]
        self.assertIsNone(resolve_followup('texture', memory))

    def test_actual_durable_photo_only_report_survives_followup_and_cannot_prompt_touching(self):
        memory = record({}, 'У меня только эта фотография. Воду после дождя не проверял, анализов нет.')
        self.assertEqual(memory['access']['status'], 'photo_only')
        memory = record(memory, 'Другой информации нет.')
        self.assertEqual(choose_followup('texture', memory), '')
        self.assertEqual(choose_followup('material', memory), '')
        self.assertEqual(choose_followup('profile', memory), '')
        self.assertEqual(choose_followup('water_regime', memory), '')

    def test_legacy_model_assignments_do_not_grant_access_or_supply_durable_facts(self):
        legacy = {'access': {'status': 'on_site'},
                  'fields': {'location': {'status': 'known', 'quote': 'Invented location'}}}
        self.assertEqual(choose_followup('texture', legacy), FOLLOWUPS['location'].text)
        self.assertEqual(choose_followup('texture', None), FOLLOWUPS['location'].text)

    def test_visual_water_color_and_structure_do_not_answer_different_physical_fields(self):
        memory = evidence(fields={
            'surface_water': {'status': 'known', 'quote': 'На фото вода'},
            'surface_color': {'status': 'known', 'quote': 'Тёмная земля'},
            'soil_structure': {'status': 'known', 'quote': 'Видны комочки'},
            'texture_description': {'status': 'known', 'quote': 'На фото похожа на глину'},
        })
        for code in ('water_regime', 'material', 'texture'):
            self.assertEqual(choose_followup(code, memory), FOLLOWUPS[code].text)
        self.assertIn('без копки', FOLLOWUPS['profile'].text)
        self.assertIn('уже есть открытый срез', FOLLOWUPS['profile'].text)
        self.assertEqual(FOLLOWUPS['texture'].minutes, 2)
        self.assertEqual(FOLLOWUPS['material'].minutes, 1)
        self.assertTrue(all(followup.cost_rub == 0 for followup in FOLLOWUPS.values()))


if __name__ == '__main__':
    unittest.main()
