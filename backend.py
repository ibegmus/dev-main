import json
from http.server import HTTPServer, BaseHTTPRequestHandler

ANSWERS = {
    "грузия": """Почва из Грузии появилась в коллекции музея в **1980 году**.

**Из контекста музея:**
**Место происхождения:** Монолит был взят в Грузии, в городе Махарадзе (регион Аджария), на чайной плантации.

**Исторический контекст:** Образец был заготовлен в период проведения Всесоюзного съезда в Тбилиси.

**Особенности и интерес:** Этот монолит является **самым крайним** в коллекции музея. История его доставки уникальна: из-за сложности транспортировки и необычности груза окружающие удивлялись тому, что из Грузии везут именно землю, а не привычные фрукты или вино. Директор института тогда твердо заявил, что доставка монолита — это их задача, и успешно организовал перевозку.

**Из научных статей (дополнение):**
**Термин «Почвы природных зон»:** Поскольку в музее экспозиция называется именно так, важно понимать, что это означает. В почвоведении «природные зоны» — это совокупность природных условий (климата, рельефа, растительности и типа материнских пород), которые определяют характер формирования почв. Почва из Грузии представляет собой пример почвообразования в специфических условиях южных регионов.""",

    "колонка": """**Геологическая колонка** — это графическое изображение вертикального разреза геологических отложений.

Она показывает:
- Последовательность слоев горных пород
- Их возраст и мощность
- Литологический состав

В контексте музея почвоведения, геологическая колонка помогает понять, как формировались почвы в разных регионах на протяжении тысячелетий.""",

    "хранить": """**Правильное хранение монолитов** включает несколько ключевых аспектов:

1. **Температурный режим:** Постоянная температура 18-22°C, без резких перепадов.

2. **Влажность:** Относительная влажность воздуха 40-60%.

3. **Освещение:** Избегать прямых солнечных лучей.

4. **Защита от вибраций:** Монолиты должны быть установлены на устойчивых подставках.

5. **Регулярный осмотр:** Не реже одного раза в год.""",

    "default": """Это интересный вопрос! К сожалению, в моей базе данных пока нет точного ответа на него.

Попробуйте спросить о:
- Почве из Грузии
- Геологической колонке
- Правилах хранения монолитов"""
}

class Handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = json.dumps({"status": "server is running"}, ensure_ascii=False)
        self.wfile.write(response.encode('utf-8'))

    def do_POST(self):
        if self.path != '/ask':
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = json.dumps({"error": "Not found"}, ensure_ascii=False)
            self.wfile.write(response.encode('utf-8'))
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)

        question_text = data.get('text', '').lower()

        answer = ANSWERS["default"]

        if "грузи" in question_text or "грузин" in question_text:
            answer = ANSWERS["грузия"]
        elif "колонк" in question_text:
            answer = ANSWERS["колонка"]
        elif "хран" in question_text or "монолит" in question_text:
            answer = ANSWERS["хранить"]

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        response = json.dumps({"answer": answer}, ensure_ascii=False)
        self.wfile.write(response.encode('utf-8'))

    def log_message(self, format, *args):
        print(f"[{self.client_address[0]}] {args[0]}")

if __name__ == '__main__':
    print("=" * 50)
    print("СЕРВЕР ЗАПУЩЕН НА http://127.0.0.1:8000")
    print("=" * 50)
    print()
    server = HTTPServer(('127.0.0.1', 8000), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен")
        server.server_close()