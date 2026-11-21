import base64
import io
import os
from PIL import Image, ImageOps, UnidentifiedImageError

import fitz  # pymupdf
from pypdf import PdfReader
from docx import Document as DocxDocument
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from model_runtime import keep_alive_seconds
from performance import invoke_chat

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
MAX_SIZE = 10 * 1024 * 1024  # 10 МБ
MAX_SCAN_PAGES = 10          # сколько страниц скана распознаём (каждая — отдельный запрос к модели)
MAX_PHOTO_SIZE = 5 * 1024 * 1024
MAX_PHOTO_PIXELS = 20_000_000

vision_llm = ChatOllama(
    model=os.getenv("OLLAMA_VISION_MODEL", "gemma4:12b-it-qat"),
    temperature=0.0, num_predict=1500, reasoning=False,
    keep_alive=keep_alive_seconds(), client_kwargs={"timeout": 240.0},
)


def decode_photo(encoded: str) -> bytes:
    """Validate the actual image and normalize orientation, size and MIME to JPEG."""
    if len(encoded) > 4 * ((MAX_PHOTO_SIZE + 2) // 3):
        raise OverflowError("Фото слишком большое (максимум 5 МБ).")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Не удалось прочитать изображение.") from exc
    if len(data) > MAX_PHOTO_SIZE:
        raise OverflowError("Фото слишком большое (максимум 5 МБ).")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Поддерживаются фотографии JPEG, PNG и WebP.")
            if image.width * image.height > MAX_PHOTO_PIXELS:
                raise ValueError("Разрешение фото слишком большое (максимум 20 мегапикселей).")
            image.load()
            image = ImageOps.exif_transpose(image)
            image.thumbnail((1600, 1600))
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGB", image.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85)
            return output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Файл повреждён или не является фотографией.") from exc

IMAGE_PROMPT = (
    "Перепиши весь текст с изображения дословно. "
    "Если текста нет, кратко опиши, что изображено (например, почвенный разрез, монолит, схема). "
    "Ничего не выдумывай. Если что-то неразборчиво, пропусти это."
)


def image_to_text(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(content=[
        {"type": "text", "text": IMAGE_PROMPT},
        {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
    ])
    return invoke_chat(vision_llm, [message], "scan").content.strip()


def scan_pdf_to_text(data: bytes) -> str:
    """Для PDF без текстового слоя: рисуем страницы и читаем их моделью."""
    pdf = fitz.open(stream=data, filetype="pdf")
    parts = []
    for page_number in range(min(len(pdf), MAX_SCAN_PAGES)):
        png = pdf[page_number].get_pixmap(dpi=150).tobytes("png")
        parts.append(image_to_text(png))
    return "\n\n".join(parts)


def extract_text(filename: str, data: bytes) -> str:
    """Достаёт текст из файла. Бросает ValueError, если формат не поддерживается."""
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        if len(text.strip()) < 50:          # похоже на скан
            text = scan_pdf_to_text(data)

    elif ext == ".docx":
        doc = DocxDocument(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        text = "\n".join(parts)

    elif ext in (".txt", ".md"):
        text = data.decode("utf-8", errors="ignore")

    elif ext in IMAGE_EXT:
        mime = "image/png" if ext == ".png" else "image/jpeg"
        text = image_to_text(data, mime)

    else:
        raise ValueError("Неподдерживаемый формат")

    return text.strip()

PHOTO_PROMPT = (
    "Опиши, что видно на фото, с точки зрения почвоведения: цвет, слои (горизонты), "
    "структура, растительность, если они видны. Описывай только то, что видно, "
    "и не называй точный тип почвы. Если на фото нет почвы, просто кратко скажи, что на нём. "
    "Не более 5 предложений."
)


def describe_photo(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(content=[
        {"type": "text", "text": PHOTO_PROMPT},
        {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
    ])
    description = invoke_chat(vision_llm, [message], "photo").content.strip()
    if not description:
        raise RuntimeError("The vision model returned an empty description")
    return description
