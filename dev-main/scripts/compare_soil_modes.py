"""Compare the answer stage locally; never pull models or send data to a new service."""
import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests/fixtures/soil_ablation"
MODES = ("native", "prompt_only", "with_reference")
ALL_MODES = MODES + ("evidence_card",)
LABELS = {"native": "Без системного промпта сайта",
          "prompt_only": "Промпт сайта без справочника",
          "with_reference": "Промпт сайта со справочником",
          "evidence_card": "Эксперимент: та же справка и карточка слов посетителя"}


def load_cases(ids):
    manifest = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
    result = []
    for case in manifest:
        if case["id"] not in ids:
            continue
        data = (FIXTURES / case["image"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != case["sha256"]:
            raise ValueError("Fixture image checksum mismatch")
        result.append((case, data))
    if {case["id"] for case, _ in result} != set(ids):
        raise ValueError("Unknown case")
    return result


def messages_for(mode, question, image):
    import soil_answer
    if mode not in ALL_MODES:
        raise ValueError("Unknown comparison mode")
    evidence = None
    if mode == "evidence_card":
        from soil_evidence import record
        evidence = record({}, question)
    messages = soil_answer.build_messages(
        question, image=image, include_reference=mode in {"with_reference", "evidence_card"},
        user_evidence=evidence)

    # Keep the entire user message, pixels and wrapper identical. The native
    # control differs from prompt_only only by the missing site system prompt.
    return messages[1:] if mode == "native" else messages


def record_response(model, messages, *, clock=time.perf_counter):
    import soil_answer
    start = clock()
    try:
        response = model.invoke(messages)
        text = soil_answer.final_text(response)
        row = {"status": "done", "answer": text,
               "words": len(re.findall(r"\S+", text))}
        metadata = response.response_metadata or {}
        row["ollama_metrics"] = {
            name: metadata[name]
            for name in ("total_duration", "load_duration", "prompt_eval_duration",
                         "eval_duration", "prompt_eval_count", "eval_count")
            if isinstance(metadata.get(name), (int, float))
            and not isinstance(metadata[name], bool)
            and 0 <= metadata[name] < 1e20
        }
    except Exception as error:
        # Provider exceptions may contain credentials or private request text.
        row = {"status": "error", "error_type": type(error).__name__}
    row["elapsed_s"] = clock() - start
    return row


def render_report(report):
    esc = lambda value: html.escape(str(value), quote=True)
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
             'img-src data:; style-src \'unsafe-inline\'; base-uri \'none\'">',
             '<title>TerraMind: сравнение режимов Ollama</title>',
             '<style>body{max-width:1100px;margin:24px auto;padding:0 16px;font:16px/1.5 system-ui;'
             'color:#25372c}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;'
             'padding:14px;background:#f1f4ef}img{max-height:360px;max-width:100%}'
             'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;'
             'padding:8px;text-align:left}.scroll{overflow:auto}code{overflow-wrap:anywhere}</style>',
             '<h1>Где возникает ошибка: модель, промпт или справочник?</h1>',
             '<p>Одни фотографии, одинаковые сведения посетителя и настройки модели. '
             'В исходных трёх режимах меняются системный промпт и справочные материалы. '
             'Это сравнение ответного этапа, без классификатора, фотоописателя, '
             'очереди и внешнего поиска. Веса не скачиваются.</p>',
             '<p>У каждого режима один запрос со всеми известными фактами, без истории '
             'ассистента. Пользовательское сообщение у native и prompt_only идентично; '
             'различается только наличие системной инструкции сайта. Это не сравнение '
             'разных интерфейсов Ollama. Изображение нормализовано тем же способом, что при загрузке на сайт. '
             'Метки источников не передаются модели. Подписи фотографий не заменяют '
             'экспертную проверку; разные классификации почв не равнозначны автоматически.</p>',
             '<p>Необязательный evidence_card добавляет к with_reference карточку, извлечённую '
             'только из слов посетителя: заявленные признаки, доступность и неизвестные свойства. '
             'Это экспериментальный контекст, не экспертные данные. Промпт, справка и фото '
             'сохраняются. Этот одиночный запрос не проверяет память многоходовой беседы.</p>',
             '<p>Один прогон не устанавливает причину ошибок окончательно. Нужны '
             'повторяемость, несколько образцов и экспертная оценка. Длительность '
             'может зависеть от первого запуска, прогрева и другой нагрузки.</p>',
             '<h2>Условия</h2><pre>' + esc(json.dumps(report["settings"], ensure_ascii=False, indent=2)) + '</pre>']
    for item in report["cases"]:
        source = item["source"]
        parts += [f'<h2>Образец {source["id"]:02d}</h2>',
                  f'<img alt="Фото образца" src="data:image/jpeg;base64,{item["image_base64"]}">',
                  '<p>Подпись источника: ' + esc(source["reference_group"]) + '. '
                  f'<a href="{esc(source["source_url"])}">Источник</a>. '
                  + esc(source["author"]) + '. '
                  f'<a href="{esc(source["license_url"])}">{esc(source["license"])}</a>.</p>',
                  '<p>Подготовка исходного файла: ' + esc(source["preprocessing"]) + '</p>',
                  '<p>SHA-256 нормализованного изображения: <code>' + item["normalized_sha256"] + '</code>.</p>',
                  '<details><summary>Данные посетителя, одинаковые во всех режимах</summary><pre>' +
                  esc(source["question"]) + '</pre></details>']
        for row in item["runs"]:
            parts.append(f'<h3>{esc(LABELS[row["mode"]])}, повтор {row["repeat"]}</h3>')
            parts.append(f'<p>{row["elapsed_s"]:.2f} с; состояние: {esc(row["status"])}; '
                         f'слов: {row.get("words", "—")}.</p>')
            metrics = row.get("ollama_metrics", {})
            stages = [(label, metrics[name] / 1e9) for name, label in (
                ("load_duration", "Загрузка"), ("prompt_eval_duration", "Обработка входа"),
                ("eval_duration", "Генерация")) if name in metrics]
            if stages:
                parts.append('<p>' + '; '.join(f'{label}: {seconds:.3f} с'
                                               for label, seconds in stages) + '</p>')
            parts.append('<pre>' + esc(row.get("answer", row.get("error_type", ""))) + '</pre>')
            parts.append('<details><summary>Вход и метрики Ollama</summary><pre>' +
                         esc(json.dumps({key: value for key, value in row.items()
                                         if key not in {"answer"}}, ensure_ascii=False, indent=2)) + '</pre></details>')
    parts += ['<details><summary>Системный промпт и справочник эксперимента</summary><pre>',
              esc(report["system_prompt"]), '\n\nСПРАВОЧНИК:\n', esc(report["reference"]), '</pre></details></html>']
    return ''.join(parts)


def evaluate(cases, output, model, *, repeats=1, modes=MODES):
    from files import decode_photo
    import soil_answer
    modes = tuple(dict.fromkeys(modes))
    if not modes or any(mode not in ALL_MODES for mode in modes):
        raise ValueError("Invalid experimental modes")
    reference = soil_answer.reference_context()
    if not reference:
        raise ValueError("Soil reference unavailable; comparison would not isolate its effect")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"settings": {"model": model.model, "reasoning": model.reasoning,
                           "num_ctx": model.num_ctx, "num_predict": model.num_predict,
                           "temperature": model.temperature, "keep_alive": model.keep_alive, "repeats": repeats,
                           "modes": list(modes), "planned_calls": len(cases) * len(modes) * repeats,
                           "time_zone": "UTC+05:00", "mode_order": "rotated across cases/repeats"},
              "system_prompt": soil_answer.PROMPT, "reference": reference,
              "cases": []}

    def save():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "report.html").write_text(render_report(report), encoding="utf-8")

    save()
    for index, (case, raw) in enumerate(cases):
        image = decode_photo(base64.b64encode(raw).decode("ascii"))
        item = {"source": case, "normalized_sha256": hashlib.sha256(image).hexdigest(),
                "image_base64": base64.b64encode(image).decode("ascii"), "runs": []}
        report["cases"].append(item)
        for repeat in range(repeats):
            offset = (index + repeat) % len(modes)
            order = modes[offset:] + modes[:offset]
            for mode in order:
                print(f'case={case["id"]:02d} repeat={repeat+1} mode={mode}', flush=True)
                messages = messages_for(mode, case["question"], image)
                # Hash the actual complete message content, including the pixels,
                # without serializing provider clients, environment or reasoning.
                payload = [{"role": m.type, "content": m.content} for m in messages]
                row = {"mode": mode, "repeat": repeat + 1,
                       "started_at": datetime.now(timezone(timedelta(hours=5))).isoformat(),
                       "messages_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                                                     sort_keys=True).encode()).hexdigest()}
                if mode == "evidence_card":
                    # Save exactly the added prefix sent in this call for review.
                    baseline = messages_for("with_reference", case["question"], image)
                    full_text = messages[-1].content[0]["text"]
                    base_text = baseline[-1].content[0]["text"]
                    if not full_text.endswith(base_text):
                        raise ValueError("Experimental context changed the baseline input")
                    row["added_evidence_context"] = full_text[:-len(base_text)].rstrip()
                row.update(record_response(model, messages))
                item["runs"].append(row)
                save()
                print(f'  {row["status"]} {row["elapsed_s"]:.2f}s words={row.get("words", 0)}', flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, action="append", choices=(1, 2, 6, 7, 8))
    parser.add_argument("--mode", choices=ALL_MODES, action="append", dest="modes",
                        help="Repeat to select modes; default: native, prompt_only, with_reference")
    parser.add_argument("--all", action="store_true", help="All five samples instead of cases 01 and 06")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeats not in range(1, 4):
        parser.error("--repeats must be between 1 and 3")
    if args.all and args.case:
        parser.error("Choose --all or --case")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import soil_answer
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    destination = args.output or ROOT / "data/evaluations" / ("soil-ablation-" + stamp)
    cases = load_cases([1, 2, 6, 7, 8] if args.all else args.case or [1, 6])
    report = evaluate(cases, destination, soil_answer.vision_llm, repeats=args.repeats,
                      modes=args.modes or MODES)
    print("Report: " + str(destination / "report.html"))
    failures = sum(row["status"] != "done" for item in report["cases"] for row in item["runs"])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
