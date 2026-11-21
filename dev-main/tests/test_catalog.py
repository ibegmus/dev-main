"""Content/link integrity checks for the reviewed photograph import."""
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import unittest

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from scripts.build_catalog import detail, outputs

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'static'
DATA = json.loads((STATIC / 'data/soils.json').read_text(encoding='utf-8'))


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        value = attrs.get('href') if tag in ('a', 'link') else attrs.get('src')
        if value and value.startswith('/'):
            self.paths.append(value.split('#')[0].split('?')[0] or '/')


class CatalogTests(unittest.TestCase):
    def test_every_uploaded_filename_is_accounted_for_and_original_bytes_preserved(self):
        expected = {f'photo_{n}_2026-10-31_11-09-{batch}.jpg'
                    for batch, count in [('19', 10), ('42', 86)]
                    for n in range(1, count + 1)}
        observed = []
        ids = []
        for soil in DATA['soils']:
            ids.append(soil['id'])
            for role, filename in soil['source_files'].items():
                observed.append(filename)
                image = STATIC / soil['images'][role]['src'].lstrip('/')
                self.assertEqual(sha256(image.read_bytes()).hexdigest(), soil['source_sha256'][role])
            label_no = int(soil['source_files']['label'].split('_')[1])
            photo_no = int(soil['source_files']['monolith'].split('_')[1])
            self.assertEqual(label_no % 2, 1)
            self.assertEqual(photo_no, label_no + 1)
        self.assertEqual(set(observed), expected)
        self.assertEqual(len(observed), 96)
        self.assertEqual(ids, [f'{i:03}' for i in range(1, 49)])

    def test_duplicate_photographs_are_never_presented_as_confirmed_pairs(self):
        by_hash = defaultdict(list)
        for soil in DATA['soils']:
            by_hash[soil['source_sha256']['monolith']].append(soil)
        duplicates = [group for group in by_hash.values() if len(group) > 1]
        self.assertEqual([[s['id'] for s in g] for g in duplicates], [['031', '041']])
        for group in duplicates:
            for soil in group:
                self.assertEqual(soil['image_status'], 'needs_review')
                self.assertTrue(any('совпадают побайтно' in n for n in soil['notes']))

    def test_source_discrepancies_and_scale_values_remain_explicit(self):
        by_id = {s['id']: s for s in DATA['soils']}
        georgia = by_id['029']
        self.assertEqual(next(r['value'] for r in georgia['facts'] if r['label'] == 'Год отбора монолита'), '1981')
        self.assertIn('1980', ' '.join(georgia['notes']))
        # These are printed inconsistencies, not gaps to "repair" by guessing.
        self.assertEqual(by_id['033']['horizons'][1]['depth_cm'], '3–50')
        self.assertEqual(by_id['041']['horizons'][3]['depth_cm'], '37–78')
        for soil in DATA['soils']:
            if soil['source_kind'] == 'profile_chart':
                self.assertTrue(soil['chart_scales'])
                self.assertNotIn('Гумус в горизонте A, %', [f['label'] for f in soil['facts']])

    def test_generated_content_is_current_and_all_catalog_routes_and_assets_resolve(self):
        app = FastAPI()
        app.mount('/', StaticFiles(directory=STATIC, html=True))
        with TestClient(app) as client:
            paths = set()
            for path, expected in outputs().items():
                actual = path.read_text(encoding='utf-8')
                self.assertEqual(actual, expected, f'Regenerate {path}')
                if path.suffix == '.html':
                    links = Links()
                    links.feed(actual)
                    paths.update(links.paths)
            for path in sorted(paths):
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200)
                    if path.startswith('/soils/'):
                        self.assertIn('Музейная табличка', response.text)
            self.assertEqual(client.get('/soils/999/').status_code, 404)

    def test_transcribed_strings_are_escaped_in_html(self):
        soil = deepcopy(DATA['soils'][0])
        soil['title'] = '<script>alert("sample")</script>'
        soil['notes'] = ['<img src=x onerror=alert(1)>']
        rendered = detail(soil, None, None)
        self.assertNotIn('<script>', rendered)
        self.assertNotIn('<img src=x', rendered)
        self.assertIn('&lt;script&gt;', rendered)


if __name__ == '__main__':
    unittest.main()
