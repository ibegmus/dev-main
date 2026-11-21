"""Publication boundaries, not a test of visual classification accuracy."""
import json
import unittest
from unittest.mock import Mock, patch

import httpx
from ollama import Client
from langchain_core.messages import AIMessage, HumanMessage

import soil_answer
import soil_diagnosis as diagnosis
from soil_evidence import record


def proposal(**updates):
    return {'scene': 'profile', 'visible': ['dark_color', 'water_visible'],
            'hypotheses': [{'name': 'Глеезёмы', 'basis': ['photo:water_visible']}],
            'cautions': ['water_not_regime'], 'next_question': 'water_regime', **updates}


class SoilDiagnosisTests(unittest.TestCase):
    def test_only_controlled_observations_and_tentative_names_are_published(self):
        evidence = record({}, 'У меня только фото. Обычный водный режим не знаю.')
        value = diagnosis.validate(proposal(hypotheses=[
            {'name': 'Глеезёмы с высоким содержанием органики', 'basis': ['photo:water_visible']},
            {'name': 'Торфяные почвы', 'basis': ['photo:water_visible', 'user:invented']},
        ]), has_image=True, evidence=evidence)
        result = diagnosis.render(value, evidence)
        self.assertIn('Торфяные почвы', result['answer'])
        self.assertNotIn('высоким содержанием', result['answer'])
        self.assertNotIn('user:invented', str(value))
        self.assertIn('не устанавливает постоянную сырость', result['answer'])
        self.assertIsNone(result['pending_field'])

    def test_photo_only_cannot_publish_a_tactile_request_or_claim_clay(self):
        evidence = record({}, 'Место неизвестно. У меня только это фото.')
        value = diagnosis.validate(proposal(next_question='texture', hypotheses=[
            {'name': 'Тяжёлый суглинок', 'basis': ['photo:cracks']},
            {'name': 'Солонцы', 'basis': ['photo:cracks']},
        ], visible=['cracks'], cautions=['shape_not_texture']), has_image=True, evidence=evidence)
        result = diagnosis.render(value, evidence)
        self.assertIn('Солонцы', result['answer'])
        self.assertNotIn('Тяжёлый суглинок', result['answer'])
        self.assertNotIn('скатать', result['answer'])
        self.assertIsNone(result['pending_field'])

    def test_unknown_visual_details_and_fake_sources_are_discarded_without_gating_names(self):
        evidence = record({}, 'Свердловская область.')
        value = diagnosis.validate(proposal(hypotheses=[
            {'name': 'Ржавозёмы', 'basis': ['photo:water_visible', 'user:location']},
            {'name': 'Dystric Gleysol', 'basis': ['user:location']},
        ]), has_image=False, evidence=evidence)
        self.assertEqual(value['visible'], [])
        self.assertEqual([row['basis'] for row in value['hypotheses']], [['user:location'], ['user:location']])
        self.assertEqual(len(value['hypotheses']), 2)
        self.assertNotIn('На фото', diagnosis.render(value, evidence)['answer'])

    def test_arbitrary_explanations_and_injected_sentences_never_enter_answer(self):
        for value in (proposal(explanation='Почва богата гумусом'),
                      proposal(visible=['high_humus']),
                      proposal(next_question='Buy expensive meter')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnosis.validate(value, has_image=True, evidence={})
        value = diagnosis.validate(proposal(hypotheses=[
            {'name': 'Глеезём. Постоянно заболочен', 'basis': []},
        ]), has_image=True, evidence={})
        self.assertFalse(value['hypotheses'])

    def test_complete_json_fence_is_accepted_but_arbitrary_text_is_not(self):
        result = diagnosis.validate('```json\n' + json.dumps(proposal()) + '\n```', has_image=True, evidence={})
        self.assertTrue(result['hypotheses'])
        with self.assertRaises(ValueError):
            diagnosis.validate('Почва богата гумусом.\n' + json.dumps(proposal()), has_image=True, evidence={})

    def test_property_predicates_cannot_be_smuggled_through_a_hypothesis_name(self):
        for name in ('Чернозёмы много гумуса', 'Глеезёмы низкая водопроницаемость',
                     'Почва бедна азотом', 'Gleysol rich humus', 'Песчаный подзол'):
            value = diagnosis.validate(proposal(hypotheses=[{'name': name, 'basis': ['photo:dark_color']}]),
                                       has_image=True, evidence={})
            with self.subTest(name=name):
                self.assertFalse(value['hypotheses'])
        for name in ('Аллювиальные серогумусовые', 'Ржавозёмы', 'Dystric Gleysol', 'Солонцы', 'Подбуры', 'Ranker'):
            self.assertTrue(diagnosis.taxonomic_name(name), name)

    def test_text_only_answer_does_not_claim_to_inspect_an_unavailable_photo(self):
        value = diagnosis.validate(proposal(scene='other', hypotheses=[],
            cautions=['water_not_regime', 'profile_missing'], next_question='none'), has_image=False, evidence={})
        answer = diagnosis.render(value, {})['answer']
        self.assertNotIn('снимк', answer)
        self.assertNotIn('вод', answer)

    def test_image_limitations_require_the_relevant_observation(self):
        value = diagnosis.validate(proposal(visible=['dark_color', 'pale_band'],
            cautions=['color_not_organic', 'shape_not_texture']), has_image=True, evidence={})
        answer = diagnosis.render(value, {})['answer']
        self.assertIn('Тёмный цвет сам по себе', answer)
        self.assertNotIn('трещин', answer)

    def test_real_adapter_sends_one_short_schema_request_with_pixels_and_history(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, json={'model': body['model'], 'done': True,
                'message': {'role': 'assistant', 'content': json.dumps(proposal()),
                            'thinking': 'PRIVATE REASONING'}})

        client = Client(transport=httpx.MockTransport(respond))
        evidence = record({}, 'Не знаю, вода постоянная или после дождя. У меня только фото.')
        with patch.object(soil_answer.vision_llm, '_client', client):
            result = soil_answer.reply('Что за почва?', image=b'jpeg', user_evidence=evidence,
                                      history=[HumanMessage(content='Мой участок'), AIMessage(content='Много гумуса')])
        body, = requests
        self.assertFalse(body['think'])
        self.assertEqual(body['format'], diagnosis.SCHEMA)
        self.assertIn('"next_question"', body['messages'][0]['content'])
        self.assertEqual(body['options']['num_predict'], 512)
        self.assertEqual([row['role'] for row in body['messages']], ['system', 'user', 'assistant', 'user'])
        self.assertTrue(body['messages'][-1]['images'])
        self.assertNotIn('PRIVATE', result['answer'])
        self.assertNotIn('Много гумуса', result['answer'])
        self.assertIsNone(result['pending_field'])

    def test_malformed_final_answer_never_falls_back_to_unconstrained_properties(self):
        with patch.object(soil_answer, 'text_llm', Mock(invoke=Mock(return_value=AIMessage(
                content='Это тяжёлая глина с плохим дренажем.')))):
            result = soil_answer.reply('Определи почву')
        self.assertNotIn('глина', result['answer'])
        self.assertNotIn('дренаж', result['answer'])
        self.assertIsNone(result['pending_field'])

    def test_invalid_short_response_gets_one_bounded_thinking_retry(self):
        client = Mock(invoke=Mock(side_effect=[AIMessage(content=''), AIMessage(content=json.dumps(proposal()))]))
        with patch.object(soil_answer, 'vision_llm', client):
            result = soil_answer.reply('Какая это почва?', image=b'jpeg')
        first, retry = client.invoke.call_args_list
        self.assertFalse(first.kwargs['reasoning'])
        self.assertEqual(first.kwargs['format'], diagnosis.SCHEMA)
        self.assertTrue(retry.kwargs['reasoning'])
        self.assertNotIn('format', retry.kwargs)
        self.assertEqual(retry.kwargs['options']['num_predict'], 1024)
        self.assertIs(first.args[0], retry.args[0])
        self.assertIn('Глеезёмы', result['answer'])

    def test_explicit_conclusion_keeps_previous_types_and_ends_questions(self):
        evidence = record({}, 'На фото видна вода.')
        model = Mock(invoke=Mock(return_value=AIMessage(content=json.dumps(proposal()))))
        with patch.object(soil_answer, 'vision_llm', model):
            result = soil_answer.reply('Подведи итог', image=b'jpeg', user_evidence=evidence,
                history=[AIMessage(content='Предварительная версия — Глеезёмы.')])
        self.assertIn('Глеезёмы', result['answer'])
        self.assertIsNone(result['question_code'])


if __name__ == '__main__':
    unittest.main()
