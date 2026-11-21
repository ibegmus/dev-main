"""Local Ollama photo-observer replay with one self-contained, private HTML report.

This measures vision observation calls, not site/queue latency or taxon accuracy.
Run only with images you intend to process on the configured Ollama server.
"""

import argparse
import base64
import hashlib
import html
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_case(path, analyze):
    from files import MAX_PHOTO_SIZE, decode_photo
    from soil_photo import eligible

    started = time.perf_counter()
    model_started = None
    result = {"file": path.name, "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_PHOTO_SIZE + 1)
        if len(raw) > MAX_PHOTO_SIZE:
            raise OverflowError("Photo size limit")
        image = decode_photo(base64.b64encode(raw).decode())
        result["original_sha256"] = hashlib.sha256(raw).hexdigest()
        result["model_image_sha256"] = hashlib.sha256(image).hexdigest()
        result["image_base64"] = base64.b64encode(image).decode()
        model_started = time.perf_counter()
        result["prepare_s"] = model_started - started
        observed = analyze(image)
        result["vision_call_s"] = time.perf_counter() - model_started
        result["description"] = observed["description"]
        result["observation"] = observed["observation"]
        result["eligible_for_confirmation"] = eligible(observed["observation"])
        result["status"] = "structured" if observed["observation"] else "unstructured"
    except Exception as error:
        # Exceptions may contain endpoint/proxy credentials. Keep only the type.
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        if model_started is not None:
            result["vision_call_s"] = time.perf_counter() - model_started
    result["total_s"] = time.perf_counter() - started
    return result


def render_report(metadata, cases):
    escape = lambda value: html.escape(str(value), quote=True)
    rows = []
    for index, case in enumerate(cases, 1):
        image = (
            '<img alt="Снимок, переданный модели" src="data:image/jpeg;base64,'
            + case["image_base64"]
            + '">'
            if case.get("image_base64")
            else ""
        )
        record = {key: value for key, value in case.items() if key != "image_base64"}
        rows.append(
            "<article><h2>"
            + str(index)
            + ". "
            + escape(case["file"])
            + "</h2>"
            + image
            + "<pre>"
            + escape(json.dumps(record, ensure_ascii=False, indent=2))
            + "</pre></article>"
        )
    return (
        """<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'">
<title>TerraMind — проверка наблюдений по фото</title>
<style>body{font:16px/1.5 system-ui;margin:24px auto;padding:0 16px;max-width:1000px;color:#183329}
article{border-top:1px solid #bbc9c2;margin-top:32px;padding-top:16px}img{max-width:100%;max-height:600px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f5f1;padding:16px;border-radius:8px}
h1{font-size:28px}h2{font-size:21px}</style><h1>Проверка наблюдений по фото</h1>
<p>Прямые последовательные вызовы фотоописателя Ollama. Время включает загрузку модели,
если она потребовалась. Это не время сайта, очереди или полного диалога и не оценка
точности классификации почв. Экспертная разметка не подставляется в модель.
Признаки ниже — наблюдения ИИ: для использования в беседе их должен подтвердить посетитель.
В отчёт встроены изображения, отправленные модели. Он может содержать личные данные;
храните и передавайте его так же, как исходные фотографии.</p><pre>"""
        + escape(json.dumps(metadata, ensure_ascii=False, indent=2))
        + "</pre>"
        + "".join(rows)
        + "</html>"
    )


def evaluate(paths, output, analyze, metadata):
    if output.exists():
        raise FileExistsError("Choose a new report filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    for path in paths:
        cases.append(run_case(path, analyze))
        # Save after every call so an interruption does not lose completed work.
        output.write_text(render_report(metadata, cases), encoding="utf-8")
        row = cases[-1]
        print(
            f"{len(cases)}/{len(paths)}: {row['status']}; total={row['total_s']:.3f}s",
            flush=True,
        )
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Проверить фотоописатель на локальных фото и сохранить единый HTML-отчёт."
    )
    parser.add_argument(
        "images",
        type=Path,
        help="Папка с JPG/PNG/WebP или отдельный файл; имена не передаются модели.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Максимум снимков, по умолчанию 10 (от 1 до 100).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "logs"
        / (
            "photo-observations-"
            + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            + ".html"
        ),
    )
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 100:
        parser.error("--limit должен быть от 1 до 100")
    if args.images.is_dir():
        paths = sorted(
            p
            for p in args.images.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )[: args.limit]
    elif args.images.is_file():
        paths = [args.images]
    else:
        parser.error("Файл или папка не найдены")
    if not paths:
        parser.error("В папке нет подходящих фотографий")
    if args.output.exists():
        parser.error("Отчёт с таким именем уже существует; задайте новый --output")
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from files import analyze_photo
    from runtime_info import runtime_info

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_info(),
        "vision_model": os.getenv("OLLAMA_VISION_MODEL", "gemma4:12b-it-qat"),
        "scope": "photo_observer_only_not_classification_accuracy",
        "requested_images": len(paths),
        "image_order": "filename_ascending",
        "execution": "sequential_local_ollama",
    }
    cases = evaluate(paths, args.output, analyze_photo, metadata)
    print("Отчёт:", args.output)
    return 1 if any(row["status"] != "structured" for row in cases) else 0


if __name__ == "__main__":
    raise SystemExit(main())
