"""A startup fingerprint for support; never expose secrets or local file paths."""
import hashlib
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def runtime_info():
    root=Path(__file__).resolve().parent
    names=['api.py','agent.py','soil_answer.py','soil_diagnosis.py','soil_evidence.py','soil_followup.py','soil_next_step.py','answer_style.py','soil_case.py','soil_interview.py','soil_dialogue.py','web_search.py','files.py','db.py','conversations.py','request_queue.py','performance.py','model_runtime.py','storage.py','soil_knowledge.py','soil_photo.py','soil_observations.py','soil_questions.py','reference_data/soil_rules.json']
    digest=hashlib.sha256()
    for name in names:
        digest.update(name.encode());digest.update((root/name).read_bytes().replace(b'\r\n',b'\n'))
    return {'service':'TerraMind','diagnosis_policy':'model-led-soil-v9-validated',
            'source_fingerprint':digest.hexdigest()[:16]}
