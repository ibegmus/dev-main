import io
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from unittest.mock import MagicMock, Mock, patch

import connection_check
import start_all


class LauncherTests(unittest.TestCase):
    def test_native_auto_protocol_command_keeps_token_out_of_arguments(self):
        command = start_all.tunnel_command('cloudflared.exe', {'CLOUDFLARE_TUNNEL_TOKEN': 'test-only-token'})
        self.assertIn('auto', command)
        self.assertEqual(command[command.index('--edge-ip-version') + 1], '4')
        self.assertEqual(command[command.index('--metrics') + 1], '127.0.0.1:20241')
        self.assertNotIn('test-only-token', ' '.join(command))
        self.assertNotIn('--token', command)

    def test_supported_protocol_override_and_invalid_configuration(self):
        command = start_all.tunnel_command('cloudflared', {'CLOUDFLARED_PROTOCOL': 'http2', 'CLOUDFLARED_EDGE_IP_VERSION': '6'})
        self.assertEqual(command[command.index('--protocol') + 1], 'http2')
        with self.assertRaises(ValueError):
            start_all.tunnel_command('cloudflared', {'CLOUDFLARED_PROTOCOL': 'invalid'})
        with self.assertRaises(ValueError):
            connection_check.metrics_port({'CLOUDFLARED_METRICS_PORT': 'http://not-a-port'})

    def test_dead_child_cannot_report_another_process_as_ready(self):
        child = Mock()
        child.poll.return_value = 1
        check = Mock(return_value=True)
        self.assertFalse(start_all.wait_until_ready(check, child, timeout=1, interval=0))
        check.assert_not_called()

    def test_waits_for_readiness_instead_of_a_fixed_startup_sleep(self):
        child = Mock()
        child.poll.return_value = None
        check = Mock(side_effect=[False, False, True])
        self.assertTrue(start_all.wait_until_ready(check, child, timeout=1, interval=0))

    def test_relay_drains_all_messages_and_redacts_secrets(self):
        stream = io.StringIO(('INF token=test-only-token proxy=https://name:password@proxy.invalid\n') * 5000)
        output = io.StringIO()
        start_all.relay_logs(stream, output, 'test-only-token')
        self.assertEqual(output.getvalue().count('\n'), 5000)
        self.assertNotIn('test-only-token', output.getvalue())
        self.assertNotIn('name:password', output.getvalue())

    def test_missing_token_does_not_launch_any_process(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(start_all, 'PROJECT_DIR', Path(directory)), \
             patch.dict(os.environ, {}, clear=True), patch.object(start_all, 'load_dotenv'), \
             patch.object(start_all.subprocess, 'Popen') as spawn, patch('builtins.print'):
            self.assertEqual(start_all.main(), 1)
            self.assertIn('Не задан CLOUDFLARE_TUNNEL_TOKEN', (Path(directory) / 'logs/startup.log').read_text(encoding='utf-8'))
        spawn.assert_not_called()

    def run_launcher(self, directory, *, existing_api=False, connected=True,
                     api_ready=True, api_exit_code=None):
        events = []
        api = Mock()
        api.poll.return_value = api_exit_code
        api.stdout = io.StringIO('API стартует: token=test-only-token proxy=https://user:password@proxy.invalid\n')
        tunnel = Mock()
        tunnel.poll.return_value = None
        tunnel.stdout = io.StringIO('INF token=test-only-token\n')

        def load_configuration(path):
            events.append('configuration')
            os.environ['CLOUDFLARE_TUNNEL_TOKEN'] = 'test-only-token'
            os.environ['ADMIN_KEY'] = 'test-admin-key'

        def spawn_process(command, **kwargs):
            events.append('tunnel' if command[1] == 'tunnel' else 'api')
            return tunnel if command[1] == 'tunnel' else api

        readiness = [connected] if existing_api else [api_ready, connected]
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(start_all, 'PROJECT_DIR', Path(directory)), \
             patch.object(start_all, 'load_dotenv', side_effect=load_configuration), \
             patch.object(start_all, 'find_cloudflared', return_value='cloudflared.exe'), \
             patch.object(start_all, 'ensure_metrics_port_free'), \
             patch.object(start_all, 'api_ready', return_value=existing_api), \
             patch.object(start_all, 'wait_until_ready', side_effect=readiness) as wait, \
             patch.object(start_all.subprocess, 'Popen', side_effect=spawn_process) as spawn, \
             patch('builtins.input', return_value=''), patch('builtins.print') as output:
            result = start_all.main()
        return result, events, spawn, output, api, tunnel, wait

    def test_environment_is_loaded_before_children_and_owned_children_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            result, events, spawn, _, api, tunnel, wait = self.run_launcher(directory)
            self.assertEqual(result, 0)
            self.assertEqual(events, ['configuration', 'api', 'tunnel'])
            self.assertEqual(wait.call_args_list[0].kwargs['timeout'], 180)
            api_call, tunnel_call = spawn.call_args_list
            self.assertEqual(api_call.kwargs['env']['ADMIN_KEY'], 'test-admin-key')
            self.assertNotIn('CLOUDFLARE_TUNNEL_TOKEN', api_call.kwargs['env'])
            self.assertEqual(api_call.kwargs['env']['PYTHONIOENCODING'], 'utf-8')
            self.assertEqual(api_call.kwargs['env']['PYTHONUNBUFFERED'], '1')
            self.assertEqual(api_call.kwargs['stdout'], start_all.subprocess.PIPE)
            self.assertEqual(api_call.kwargs['stderr'], start_all.subprocess.STDOUT)
            self.assertNotIn('creationflags', api_call.kwargs)
            self.assertEqual(tunnel_call.kwargs['env']['TUNNEL_TOKEN'], 'test-only-token')
            self.assertEqual(api_call.kwargs['cwd'], Path(directory))
            self.assertEqual(api_call.args[0][0], start_all.sys.executable)
            self.assertNotIn('test-only-token', (Path(directory) / 'logs/cloudflared.log').read_text(encoding='utf-8'))
            startup_log = (Path(directory) / 'logs/startup.log').read_text(encoding='utf-8')
            self.assertIn('API стартует:', startup_log)
            self.assertNotIn('test-only-token', startup_log)
            self.assertNotIn('user:password', startup_log)
            api.terminate.assert_called_once()
            tunnel.terminate.assert_called_once()

    def test_existing_api_is_reused_and_not_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            _, events, spawn, _, api, tunnel, _ = self.run_launcher(directory, existing_api=True)
            self.assertEqual(events, ['configuration', 'tunnel'])
            self.assertEqual(spawn.call_count, 1)
            api.terminate.assert_not_called()
            tunnel.terminate.assert_called_once()

    def test_tunnel_timeout_is_not_announced_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, output, _, _, _ = self.run_launcher(directory, connected=False)
        messages = '\n'.join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn('Соединение туннеля пока не подтверждено', messages)
        self.assertNotIn('Туннель подключён.', messages)

    def test_api_exit_and_timeout_preserve_the_diagnostic_log(self):
        for exit_code in (7, None):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                result, events, _, _, _, _, _ = self.run_launcher(
                    directory, api_ready=False, api_exit_code=exit_code,
                )
                self.assertEqual(result, 1)
                self.assertEqual(events, ['configuration', 'api'])
                text = (Path(directory) / 'logs/startup.log').read_text(encoding='utf-8')
                self.assertIn('API стартует:', text)
                self.assertIn('кодом 7' if exit_code == 7 else 'не ответил за 3 минуты', text)
                if exit_code == 7:
                    self.assertNotIn('не ответил за 3 минуты', text)

    def test_real_child_exit_is_captured_instead_of_a_disappearing_console(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'api.py').write_text(
                'import sys\nprint("Ошибка API: модель недоступна 🚫 test-only-token", flush=True)\nsys.exit(7)\n',
                encoding='utf-8',
            )
            with patch.object(start_all, 'PROJECT_DIR', root), \
                 patch.dict(os.environ, {'CLOUDFLARE_TUNNEL_TOKEN': 'test-only-token'}), \
                 patch.object(start_all, 'load_dotenv'), \
                 patch.object(start_all, 'find_cloudflared', return_value='unused-cloudflared'), \
                 patch.object(start_all, 'ensure_metrics_port_free'), \
                 patch.object(start_all, 'api_ready', return_value=False), patch('builtins.print'):
                self.assertEqual(start_all.main(), 1)
            text = (root / 'logs/startup.log').read_text(encoding='utf-8')
            self.assertIn('Ошибка API: модель недоступна 🚫', text)
            self.assertIn('кодом 7', text)
            self.assertNotIn('test-only-token', text)
            self.assertNotIn('не ответил за 3 минуты', text)


class LoopbackChecksTests(unittest.TestCase):
    def test_unrelated_response_is_not_mistaken_for_our_api(self):
        for body in [b'[]', b'null', b'<html>Other service</html>', b'{"paths": ["/ask", "/history"]}']:
            with self.subTest(body=body):
                self.assertFalse(connection_check.expected_api_schema(body))

    def test_public_domain_is_checked_separately_from_tunnel_connection(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = json.dumps({'paths': {'/ask': {}, '/history': {}}}).encode()
        with patch.object(connection_check, 'urlopen', return_value=response):
            self.assertEqual(connection_check.public_api_status(), (True, 'доступен'))

    def test_public_domain_denial_reports_http_status_without_error_body(self):
        denied = HTTPError('https://api.pochva-ai.tech/openapi.json', 403, 'Forbidden', {}, None)
        with patch.object(connection_check, 'urlopen', side_effect=denied):
            success, message = connection_check.public_api_status()
        self.assertFalse(success)
        self.assertIn('403', message)

    def test_public_connection_error_does_not_claim_the_tunnel_is_broken(self):
        with patch.object(connection_check, 'urlopen', side_effect=URLError('Network unavailable')):
            success, message = connection_check.public_api_status()
        self.assertFalse(success)
        self.assertNotIn('туннель', message)

    def test_http_readiness_ignores_an_unavailable_vpn_proxy(self):
        class Handler(BaseHTTPRequestHandler):
            connected = True

            def do_GET(self):
                status = 200 if self.path != '/ready' or self.connected else 503
                body = json.dumps({'paths': {'/ask': {}, '/history': {}}}).encode()
                self.send_response(status)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with patch.dict(os.environ, {'HTTP_PROXY': 'http://127.0.0.1:1', 'http_proxy': 'http://127.0.0.1:1', 'NO_PROXY': '', 'no_proxy': ''}), \
                 patch.object(connection_check, 'API_SCHEMA', f'http://127.0.0.1:{server.server_port}/openapi.json'):
                self.assertTrue(connection_check.api_ready())
                self.assertTrue(connection_check.tunnel_ready(server.server_port))
                Handler.connected = False
                self.assertFalse(connection_check.tunnel_ready(server.server_port))
            with self.assertRaises(RuntimeError):
                start_all.ensure_metrics_port_free(server.server_port)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_tcp_diagnostics_do_not_treat_timeouts_as_success(self):
        with patch.object(connection_check.socket, 'create_connection', side_effect=TimeoutError):
            self.assertIn('нет ответа', connection_check.probe_tcp('region1.v2.argotunnel.com'))
        with patch.object(connection_check.socket, 'create_connection', side_effect=socket.gaierror):
            self.assertIn('DNS', connection_check.probe_tcp('region1.v2.argotunnel.com'))
        with patch.object(connection_check.socket, 'create_connection', return_value=MagicMock()):
            self.assertEqual(connection_check.probe_tcp('region1.v2.argotunnel.com'), 'доступен')


if __name__ == '__main__':
    unittest.main()
