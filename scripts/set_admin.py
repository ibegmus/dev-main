"""Run on the server: python scripts/set_admin.py EMAIL [--revoke]."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import db

parser = argparse.ArgumentParser(description='Назначить роль существующему аккаунту локально на сервере.')
parser.add_argument('email')
parser.add_argument('--revoke', action='store_true')
args = parser.parse_args()
db.init_db()
with db.get_conn() as conn:
    count = conn.execute('UPDATE users SET role=? WHERE email=?',
                         ('user' if args.revoke else 'admin', args.email.strip().lower())).rowcount
if not count:
    sys.exit('Аккаунт не найден. Сначала зарегистрируйтесь на сайте.')
print('Роль обновлена. Обновите страницу сайта.')
