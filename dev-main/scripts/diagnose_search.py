"""Run on the actual server; prints no passwords, tokens or proxy values."""
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1]/'.env')
from web_search import search_web
from runtime_info import runtime_info
print(json.dumps(runtime_info(),ensure_ascii=False))
result=search_web('подзолистые почвы диагностические горизонты',max_results=3)
print(json.dumps({'attempts':result.get('web_attempts'), 'provider':result.get('web_provider'),
                  'error':result.get('web_error'), 'sources':result.get('web_sources')},ensure_ascii=False,indent=2))
if result.get('web_error'):raise SystemExit(1)
