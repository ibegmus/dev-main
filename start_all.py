"""Start the local API and a monitored Cloudflare tunnel."""

import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime

from dotenv import load_dotenv

from connection_check import api_ready, metrics_port, tunnel_ready

PROJECT_DIR = Path(__file__).resolve().parent
PUBLIC_URL = 'https://api.pochva-ai.tech'


def find_cloudflared():
    local = PROJECT_DIR / ('cloudflared.exe' if os.name == 'nt' else 'cloudflared')
    executable = str(local) if local.is_file() else shutil.which('cloudflared')
    if not executable:
        raise FileNotFoundError('Не найден cloudflared. Положите cloudflared.exe рядом с запуск.bat или установите cloudflared в PATH.')
    return executable


def tunnel_command(executable, environment):
    protocol = environment.get('CLOUDFLARED_PROTOCOL', 'auto').strip()
    ip_version = environment.get('CLOUDFLARED_EDGE_IP_VERSION', '4').strip()
    if protocol not in {'auto', 'quic', 'http2'}:
        raise ValueError('CLOUDFLARED_PROTOCOL: допустимы auto, quic или http2.')
    if ip_version not in {'auto', '4', '6'}:
        raise ValueError('CLOUDFLARED_EDGE_IP_VERSION: допустимы auto, 4 или 6.')
    return [executable, 'tunnel', '--protocol', protocol, '--edge-ip-version', ip_version,
            '--metrics', f'127.0.0.1:{metrics_port(environment)}', 'run']


def redact_log(line, token):
    if token:
        line = line.replace(token, '[скрыто]')
    return re.sub(r'(https?://)[^/\s@]+@', r'\1[скрыто]@', line)


def ensure_metrics_port_free(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=0.5):
            raise RuntimeError(f'Порт проверки туннеля {port} уже занят. Закройте прошлый запуск или задайте другой CLOUDFLARED_METRICS_PORT в .env.')
    except OSError:
        pass


def relay_logs(stream, output, token):
    # Always drain the pipe: repeated network errors must not block cloudflared.
    try:
        for line in stream:
            safe = redact_log(line, token)
            output.write(safe)
            output.flush()
            if ' ERR ' in safe or ' WRN ' in safe:
                print(safe.rstrip(), flush=True)
    finally:
        stream.close()


def relay_api_logs(stream, report):
    try:
        for line in stream:
            report(line.rstrip('\r\n'))
    finally:
        stream.close()


def wait_until_ready(check, process, timeout, interval=0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        if check():
            return True
        time.sleep(interval)
    return False


def stop_owned_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main():
    api_process = tunnel_process = None
    log_output = log_thread = startup_output = api_log_thread = None
    token = None
    log_lock = threading.Lock()

    def report(message):
        safe = redact_log(str(message), token)
        with log_lock:
            print(safe, flush=True)
            if startup_output is not None:
                startup_output.write(safe + '\n')
                startup_output.flush()

    try:
        log_dir = PROJECT_DIR / 'logs'
        log_dir.mkdir(exist_ok=True)
        startup_path = log_dir / 'startup.log'
        startup_output = startup_path.open('a', encoding='utf-8')
        report(f'--- Запуск {datetime.now().astimezone().isoformat(timespec="seconds")} ---')
        report(f'Журнал запуска: {startup_path}')
        # Child processes must inherit .env settings before they start.
        load_dotenv(PROJECT_DIR / '.env')
        token = os.environ.get('CLOUDFLARE_TUNNEL_TOKEN') or os.environ.get('TUNNEL_TOKEN')
        if not token:
            raise ValueError('Не задан CLOUDFLARE_TUNNEL_TOKEN. Добавьте его в локальный .env.')
        command = tunnel_command(find_cloudflared(), os.environ)
        port = metrics_port()
        ensure_metrics_port_free(port)
        report('1. Проверяем локальный сервер...')
        if not api_ready():
            api_environment = dict(os.environ)
            api_environment.pop('CLOUDFLARE_TUNNEL_TOKEN', None)
            api_environment.pop('TUNNEL_TOKEN', None)
            api_environment['PYTHONUNBUFFERED'] = '1'
            api_environment['PYTHONIOENCODING'] = 'utf-8'
            api_process = subprocess.Popen(
                [sys.executable, str(PROJECT_DIR / 'api.py')], cwd=PROJECT_DIR,
                env=api_environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1,
            )
            api_log_thread = threading.Thread(
                target=relay_api_logs, args=(api_process.stdout, report), daemon=True,
            )
            api_log_thread.start()
            report('Ожидаем ответа API. Вывод сервера отображается в этом окне и в logs/startup.log.')
            if not wait_until_ready(api_ready, api_process, timeout=180):
                exit_code = api_process.poll()
                if exit_code is not None:
                    raise RuntimeError(f'API завершился с кодом {exit_code}. Причина выше и в logs/startup.log.')
                raise RuntimeError('API не ответил за 3 минуты. Проверьте logs/startup.log и занятость порта 8000.')
        report('Локальный сервер отвечает: http://127.0.0.1:8000')

        log_output = (log_dir / 'cloudflared.log').open('a', encoding='utf-8')
        log_output.write(f'\n--- Запуск {datetime.now().astimezone().isoformat(timespec="seconds")} ---\n')
        log_output.flush()
        tunnel_process = subprocess.Popen(
            command, cwd=PROJECT_DIR, env={**os.environ, 'TUNNEL_TOKEN': token},
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
        )
        log_thread = threading.Thread(target=relay_logs, args=(tunnel_process.stdout, log_output, token), daemon=True)
        log_thread.start()
        report('2. Проверяем соединение с Cloudflare. Это может занять до 45 секунд...')
        if wait_until_ready(lambda: tunnel_ready(port), tunnel_process, timeout=45):
            report(f'Туннель подключён. Адрес сайта: {PUBLIC_URL}')
        else:
            report('Соединение туннеля пока не подтверждено. Локальный сайт продолжает работать.')
            report('Запустите «проверка_соединения.bat» без VPN. Ошибки туннеля записываются в logs/cloudflared.log.')
        input('\nНажмите Enter, чтобы остановить запущенные этим скриптом процессы сайта...')
        return 0
    except (KeyboardInterrupt, EOFError):
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        report(f'Ошибка запуска: {error}')
        return 1
    finally:
        stop_owned_process(tunnel_process)
        if log_thread:
            log_thread.join(timeout=2)
        if log_output:
            log_output.close()
        stop_owned_process(api_process)
        if api_log_thread:
            api_log_thread.join(timeout=2)
        if startup_output:
            startup_output.close()


if __name__ == '__main__':
    raise SystemExit(main())
