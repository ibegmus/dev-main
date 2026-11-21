"""Read-only checks on the computer hosting the site; no credentials are printed."""

import json
import os
from pathlib import Path
import socket
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
API_SCHEMA = 'http://127.0.0.1:8000/openapi.json'
OLLAMA_TAGS = 'http://127.0.0.1:11434/api/tags'
PUBLIC_SCHEMA = 'https://api.pochva-ai.tech/openapi.json'
EDGE_HOSTS = ('region1.v2.argotunnel.com', 'region2.v2.argotunnel.com')


def local_get(url, timeout=2):
    # These endpoints are loopback only; an inactive VPN proxy must not intercept them.
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(url), timeout=timeout) as response:
        return response.status, response.read(1024 * 1024)


def api_ready():
    try:
        status, body = local_get(API_SCHEMA)
        return status == 200 and expected_api_schema(body)
    except (OSError, URLError, ValueError):
        return False


def ollama_ready():
    try:
        status, body = local_get(OLLAMA_TAGS)
        data = json.loads(body)
        return status == 200 and isinstance(data, dict) and isinstance(data.get('models'), list)
    except (OSError, URLError, ValueError):
        return False


def expected_api_schema(body):
    try:
        data = json.loads(body)
    except ValueError:
        return False
    paths = data.get('paths') if isinstance(data, dict) else None
    return isinstance(paths, dict) and '/ask' in paths and '/history' in paths


def metrics_port(environment=None):
    environment = os.environ if environment is None else environment
    try:
        port = int(environment.get('CLOUDFLARED_METRICS_PORT', '20241'))
    except ValueError:
        raise ValueError('CLOUDFLARED_METRICS_PORT должен быть числом от 1024 до 65535.') from None
    if not 1024 <= port <= 65535:
        raise ValueError('CLOUDFLARED_METRICS_PORT должен быть числом от 1024 до 65535.')
    return port


def tunnel_ready(port):
    try:
        status, _ = local_get(f'http://127.0.0.1:{port}/ready')
        return status == 200
    except (OSError, URLError, ValueError):
        return False


def probe_tcp(host, port=7844, timeout=4):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return 'доступен'
    except socket.gaierror:
        return 'не удалось определить адрес (DNS)'
    except TimeoutError:
        return 'нет ответа за время проверки'
    except OSError:
        return 'соединение не установлено'


def public_api_status():
    # Public requests retain the system proxy and normal TLS verification.
    try:
        with urlopen(PUBLIC_SCHEMA, timeout=8) as response:
            if response.status == 200 and expected_api_schema(response.read(1024 * 1024)):
                return True, 'доступен'
            return False, 'ответ получен, но это не ожидаемый API сайта'
    except HTTPError as error:
        return False, f'домен ответил HTTP {error.code}'
    except (OSError, URLError, ValueError):
        return False, 'запрос к домену не прошёл'


def main():
    load_dotenv(PROJECT_DIR / '.env')
    print('Проверка соединения на этом компьютере. Для сравнения запустите её без VPN.\n')
    local = api_ready()
    engine = ollama_ready()
    try:
        connected = tunnel_ready(metrics_port())
    except ValueError as error:
        print(str(error))
        return 1
    print('Локальный сайт:', 'работает' if local else 'не отвечает')
    print('Ollama:', 'отвечает' if engine else 'не отвечает')
    print('Туннель Cloudflare:', 'подключён' if connected else 'подключение не подтверждено')
    public, public_message = public_api_status()
    print('Публичный API по домену:', public_message, flush=True)
    print('\nПроверка TCP-порта Cloudflare (UDP этим тестом не проверяется):')
    for host in EDGE_HOSTS:
        print(f'  {host}:7844 — {probe_tcp(host)}', flush=True)
    if not local:
        print('\nСначала запустите «запуск.bat» и проверьте окно API-сервера.')
    elif connected and public:
        print('\nЛокальный сервер, туннель и публичный API отвечают.')
    elif connected:
        print('\nТуннель подключён, но запрос по домену не прошёл успешно. Проверьте доступ к домену из этой сети, DNS и настройки сайта.')
    else:
        print('\nЛокальный сервер работает, но туннель не подключён. Подробности — в logs/cloudflared.log.')
        print('Если соединение с Cloudflare проходит только через VPN, нужно проверить сетевые ограничения или использовать другой туннель/сервер.')
    return 0 if local and connected and public else 1


if __name__ == '__main__':
    raise SystemExit(main())
