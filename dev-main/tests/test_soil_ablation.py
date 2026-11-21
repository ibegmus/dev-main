"""Validate experimental controls and reporting, not model recognition accuracy."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
from langchain_ollama import ChatOllama
from ollama import Client
from PIL import Image

from scripts import compare_soil_modes as comparison
import soil_answer


class SoilAblationTests(unittest.TestCase):
    def test_real_adapter_changes_only_prompt_and_reference_and_keeps_same_pixels(self):
        stream = io.BytesIO()
        Image.new('RGB', (300, 400), 'brown').save(stream, format='JPEG')
        raw = stream.getvalue()
        source = {'id': 1, 'image': 'case-01.jpg', 'sha256': hashlib.sha256(raw).hexdigest(),
                  'question': 'Где-то в лесу, анализов нет. Помоги определить почву.',
                  'reference_group': 'SECRET_GROUND_TRUTH', 'source_url': 'https://example.com/source',
                  'author': 'Test', 'license': 'CC BY-SA 4.0',
                  'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/',
                  'preprocessing': 'Test image'}
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, json={
                'model': body['model'], 'done': True,
                'message': {'role': 'assistant', 'content': '<script>not executed</script> Ответ.',
                            'thinking': 'PRIVATE REASONING'},
                'load_duration': 123, 'eval_count': 20})

        client = Client(transport=httpx.MockTransport(respond))
        model = ChatOllama(model='test-model', reasoning=True, num_ctx=16384,
                           num_predict=8192, temperature=.1, keep_alive=1800)
        with tempfile.TemporaryDirectory() as temp, patch.object(model, '_client', client):
            output = Path(temp) / 'run'
            result = comparison.evaluate([(source, raw)], output, model)
            self.assertEqual(len(requests), 3)
            self.assertEqual([r['messages'][0]['role'] for r in requests], ['user', 'system', 'system'])
            self.assertEqual(requests[0]['messages'][0], requests[1]['messages'][1])
            self.assertEqual(requests[1]['messages'][0], requests[2]['messages'][0])
            self.assertEqual(requests[1]['messages'][0]['content'], soil_answer.PROMPT)
            self.assertEqual(len({r['messages'][-1]['images'][0] for r in requests}), 1)
            image = base64.b64decode(requests[0]['messages'][-1]['images'][0])
            self.assertEqual(hashlib.sha256(image).hexdigest(), result['cases'][0]['normalized_sha256'])
            for index, body in enumerate(requests):
                self.assertIs(body['think'], True)
                self.assertEqual(body['options']['num_predict'], 8192)
                self.assertEqual(body['options']['num_ctx'], 16384)
                self.assertEqual(body['options']['temperature'], .1)
                text = json.dumps(body, ensure_ascii=False)
                self.assertIn(source['question'], text)
                self.assertNotIn('SECRET_GROUND_TRUTH', text)
                self.assertNotIn('https://example.com/source', text)
                self.assertEqual('https://soils.narod.ru/taxon/type/podz07.html' in text, index == 2)
                self.assertNotEqual(body.get('format'), 'json')
            report = (output / 'report.html').read_text()
            self.assertIn('&lt;script&gt;', report)
            self.assertNotIn('<script>', report)
            self.assertIn('Content-Security-Policy', report)
            self.assertIn('data:image/jpeg;base64,', report)
            self.assertNotIn('PRIVATE', report)
            self.assertNotIn('PRIVATE', (output / 'report.json').read_text())
            with self.assertRaises(FileExistsError):
                comparison.evaluate([(source, raw)], output, model)
            requests.clear()
            pair = comparison.evaluate([(source, raw)], Path(temp) / 'pair', model,
                                       modes=('with_reference', 'evidence_card'))
            self.assertEqual(len(requests), 2)
            self.assertEqual(pair['settings']['planned_calls'], 2)
            self.assertEqual(requests[0]['messages'][0], requests[1]['messages'][0])
            self.assertEqual(requests[0]['messages'][-1]['images'], requests[1]['messages'][-1]['images'])
            extra = pair['cases'][0]['runs'][1]['added_evidence_context']
            self.assertTrue(extra)
            self.assertTrue(requests[1]['messages'][-1]['content'].endswith(
                requests[0]['messages'][-1]['content']))
            self.assertIn(extra, requests[1]['messages'][-1]['content'])

    def test_all_fixture_hashes_and_queries_are_valid(self):
        cases = comparison.load_cases([1, 2, 6, 7, 8])
        self.assertEqual(len(cases), 5)
        for source, image in cases:
            with Image.open(io.BytesIO(image)) as photo:
                photo.verify()
            self.assertTrue(source['question'])
            self.assertIn('creativecommons.org', source['license_url'])

    def test_experimental_card_preserves_system_reference_and_pixels(self):
        question = 'У меня только фото. На фото видна вода. На ощупь не проверял, анализов нет.'
        baseline = comparison.messages_for('with_reference', question, b'same-image')
        candidate = comparison.messages_for('evidence_card', question, b'same-image')
        self.assertEqual(baseline[0].content, candidate[0].content)
        self.assertEqual(baseline[-1].content[1], candidate[-1].content[1])
        base_text = baseline[-1].content[0]['text']
        next_text = candidate[-1].content[0]['text']
        self.assertIn(base_text, next_text)
        self.assertIn('только фото', next_text)
        self.assertIn('Вода на одном фото не устанавливает', next_text)
        self.assertIn('На ощупь не проверял', next_text)
        # It adds no expert caption or invented measurement to the input.
        self.assertNotIn('Histosol', next_text)
        self.assertNotIn('Podzol', next_text)

    def test_real_uploaded_cases_leave_permanent_water_and_measured_texture_unknown(self):
        from soil_evidence import record
        cases = comparison.load_cases([1, 6])
        for source, _ in cases:
            evidence = record({}, source['question'])
            self.assertEqual(evidence['access']['status'], 'unknown')
            water = evidence['fields'].get('water', {})
            texture = evidence['fields'].get('texture', {})
            self.assertNotEqual(water.get('status'), 'known')
            self.assertNotEqual(texture.get('status'), 'known')
            self.assertNotEqual(evidence['fields'].get('organic_layer', {}).get('status'), 'known')

    def test_provider_error_text_and_reasoning_are_not_used_as_answer(self):
        model = Mock()
        model.invoke.side_effect = RuntimeError('PRIVATE credentials and request')
        row = comparison.record_response(model, [], clock=Mock(side_effect=[1, 4]))
        self.assertEqual(row, {'status': 'error', 'error_type': 'RuntimeError', 'elapsed_s': 3})
        self.assertNotIn('PRIVATE', json.dumps(row))

    def test_missing_reference_cannot_masquerade_as_a_reference_experiment(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
                soil_answer, 'reference_context', return_value=''):
            output = Path(temp) / 'run'
            with self.assertRaises(ValueError):
                comparison.evaluate([], output, Mock())
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
