"""Conversation presentation changes; these tests never invoke a model."""
import unittest

import soil_diagnosis as diagnosis


def evidence():
    return {'kind': 'user_soil_evidence', 'version': 1,
            'access': {'status': 'photo_only', 'quote': 'У меня только фото'},
            'fields': {'location': {'status': 'known', 'quote': 'в екб'}}}


def proposal(names=('Глеезёмы',), *, cautions=('water_not_regime',), question='landscape'):
    return diagnosis.validate({
        'scene': 'profile', 'visible': ['water_visible', 'dark_color'],
        'hypotheses': [{'name': name, 'basis': ['photo:water_visible']} for name in names],
        'cautions': list(cautions), 'next_question': question,
    }, has_image=True, evidence=evidence())


class SoilReplyRepetitionTests(unittest.TestCase):
    def test_known_location_continues_with_only_the_new_question(self):
        old = diagnosis.render(proposal(question='location'), {})['answer']
        result = diagnosis.render(proposal(cautions=('water_not_regime', 'color_not_organic')),
                                  evidence(), followup=True, previous_answers=[old])
        self.assertEqual(result['answer'], 'Если знаете, что окружает место: лес, луг, поле или сад?')
        self.assertEqual(result['pending_field'], 'landscape')

    def test_candidate_order_case_and_yo_spelling_do_not_repeat_names(self):
        old = 'Предварительные версии — Глеезёмы или Торфяные почвы. '
        old += diagnosis.CAUTIONS['water_not_regime']
        value = proposal(('торфяные почвы', 'глееземы'))
        result = diagnosis.render(value, evidence(), followup=True, previous_answers=[old])
        self.assertNotIn('Предварительн', result['answer'])
        self.assertNotIn('торфян', result['answer'])
        self.assertNotIn('Вода на снимке', result['answer'])

    def test_new_or_removed_candidate_remains_visible(self):
        old = 'Предварительная версия — Глеезёмы. ' + diagnosis.CAUTIONS['water_not_regime']
        changed = diagnosis.render(proposal(('Глеезёмы', 'Торфяные почвы')), evidence(),
                                   followup=True, previous_answers=[old])
        self.assertIn('Предварительные версии — Глеезёмы или Торфяные почвы.', changed['answer'])
        previous_pair = 'Предварительные версии — Глеезёмы или Торфяные почвы.'
        narrowed = diagnosis.render(proposal(), evidence(), followup=True, previous_answers=[previous_pair])
        self.assertIn('Предварительная версия — Глеезёмы.', narrowed['answer'])

    def test_cautions_are_deduplicated_across_question_only_turns(self):
        first = 'Предварительная версия — Глеезёмы. ' + diagnosis.CAUTIONS['water_not_regime']
        recent = [first, 'Если знаете, сюда завозили грунт или это местная земля?']
        result = diagnosis.render(proposal(cautions=('water_not_regime', 'color_not_organic'), question='none'),
                                  evidence(), followup=True, previous_answers=recent)
        self.assertNotIn(diagnosis.CAUTIONS['water_not_regime'], result['answer'])
        self.assertIn(diagnosis.CAUTIONS['color_not_organic'], result['answer'])
        self.assertNotIn('Глеезёмы', result['answer'])

    def test_same_candidates_without_a_viable_question_have_a_nonempty_closing(self):
        old = 'Предварительная версия — Глеезёмы. ' + diagnosis.CAUTIONS['water_not_regime']
        for question in ('none', 'location', 'texture'):
            with self.subTest(question=question):
                result = diagnosis.render(proposal(question=question), evidence(), followup=True,
                                          previous_answers=[old])
                self.assertIn('уже открытого среза', result['answer'])
                self.assertNotIn('Глеезёмы', result['answer'])
                self.assertNotIn('?', result['answer'])
                self.assertIsNone(result['pending_field'])

    def test_explicit_recap_shows_the_current_candidates(self):
        old = 'Предварительная версия — Глеезёмы. ' + diagnosis.CAUTIONS['water_not_regime']
        result = diagnosis.render(proposal(question='none'), evidence(), followup=True,
                                  previous_answers=[old], recap=True)
        self.assertEqual(result['answer'], 'Предварительная версия — Глеезёмы.')

    def test_old_free_prose_is_not_used_as_a_candidate_list(self):
        old = 'Скорее всего, Глеезёмы. Возможно, Глеезёмы — это наиболее вероятно.'
        result = diagnosis.render(proposal(), evidence(), followup=True, previous_answers=[old])
        self.assertIn('Предварительная версия — Глеезёмы.', result['answer'])

    def test_non_soil_or_insufficient_response_resets_the_displayed_candidate_comparison(self):
        recent = ['Предварительная версия — Глеезёмы.',
                  'Пока нельзя обоснованно назвать тип почвы по доступным признакам.']
        result = diagnosis.render(proposal(), evidence(), followup=True, previous_answers=recent)
        self.assertIn('Предварительная версия — Глеезёмы.', result['answer'])

    def test_current_abstention_never_restores_older_candidates(self):
        result = diagnosis.render(proposal((), question='none', cautions=()), evidence(),
                                  followup=True, previous_answers=['Предварительная версия — Глеезёмы.'])
        self.assertIn('Пока нельзя обоснованно назвать тип', result['answer'])
        self.assertNotIn('Глеезёмы', result['answer'])

    def test_clearing_history_for_a_new_site_or_photo_gives_the_full_first_response(self):
        result = diagnosis.render(proposal(), evidence(), previous_answers=())
        self.assertIn('Предварительная версия — Глеезёмы.', result['answer'])
        self.assertIn('На фото можно различить', result['answer'])
        self.assertIn(diagnosis.CAUTIONS['water_not_regime'], result['answer'])


if __name__ == '__main__':
    unittest.main()
