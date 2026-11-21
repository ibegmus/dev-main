import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import api
import db
from runtime_info import runtime_info

class RuntimeInfoTests(unittest.TestCase):
    def test_fingerprint_matches_running_api_without_environment_or_paths(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(db,'DB_FILE',str(Path(folder)/'app.db')),patch.object(api,'preload_reranker'),TestClient(api.app) as client:
            response=client.get('/health')
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.headers['cache-control'],'no-store')
            self.assertEqual(response.json(),runtime_info())
            self.assertEqual(set(response.json()),{'service','diagnosis_policy','source_fingerprint'})
            self.assertRegex(response.json()['source_fingerprint'],r'^[a-f0-9]{16}$')
