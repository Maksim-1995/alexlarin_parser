"""OCR всех условий по одному изображению полной страницы через Gemini.

Поддерживается двухколоночная вёрстка с номерами 1.–19. в текстовом слое.
Номера нужны для проверки состава страницы; условия читаются с изображения.
Служебные PNG, ответы API и кэши страниц сохраняются рядом с исходным PDF.

Отдельный запуск без скачивания с сайта:
    python -u recognizer.py --pdf input/trvar542.pdf
"""

import argparse
import hashlib
import json
import os
import unicodedata
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import pymupdf
from google import genai
from google.genai import types

from parsers import TASK_RE


MODEL = "gemini-3.5-flash-lite"
METHOD = "gemini_image_ocr"
DPI = 300
CACHE_VERSION = 2
TASK_NUMBERS = list(range(1, 20))

# Эти поля сохранены для чтения пригодных полных кэшей прежнего OCR.
# Новые запросы используют PAGE_PROMPT и постраничную схему ниже.

PROMPT = r"""
Перепиши полное условие одного математического задания с изображения.
Номер задания указан отдельно в запросе; верни его в task_num.
Не повторяй номер в condition. Сохрани формулировку и все подпункты.
Не исправляй опечатки исходника и не добавляй сведения от себя.
Не решай задачу и не включай ответы, бланк «Ответ: ____», колонтитулы
и общие инструкции экзамена о заполнении бланков и записи ответов.
Рисунок будет сохранён отдельно: не придумывай его словесное описание.

Сохрани исходную математическую запись, не выполняй преобразований.
Формулы записывай в LaTeX и заключай в $...$.
Особенно внимательно прочитай дробные черты, корни, показатели степеней,
основания логарифмов, скобки и границы интервалов.
Корень должен относиться именно к выражению под его чертой.
Не переноси выражения из одной строки системы в другую.
Не используй блоки Markdown или пояснения вне JSON.
Если фрагмент нельзя прочитать уверенно, поставь [НЕРАЗБОРЧИВО].
Верни список из одной записи: task_num и condition.
"""

PAGE_CACHE_VERSION = 1
AGGREGATE_CACHE_VERSION = 3
PAGE_METHOD = "gemini_page_image_ocr"
REQUEST_TIMEOUT_MS = 120_000
MAX_OUTPUT_TOKENS = 16384
PAGE_PROMPT = r"""
Перепиши полные условия всех математических заданий с изображения страницы.
Номер страницы указан отдельно; верни его в page_num.
Страница имеет две колонки. Читай задания в обеих колонках, ориентируясь
на их напечатанные номера. Каждое задание верни один раз: task_num и condition.
Не включай номер задания в condition. Сохраняй все пункты А, Б, В.
Не исправляй опечатки исходника и не добавляй сведения от себя.
Не решай задачи и не генерируй ответы. Исключи титульный и служебный текст,
бланки «Ответ: ____», колонтитулы и инструкции экзамена.
Рисунки сохраняются отдельно: не придумывай их словесное описание.

Сохрани исходную математическую запись без преобразований.
Формулы записывай в LaTeX и заключай в $...$.
Внимательно прочитай дроби, корни, степени, основания логарифмов, скобки,
границы интервалов и все строки систем. Корень относится только к выражению
под его чертой. Не переноси части формул из соседней колонки или задания.
Если фрагмент нельзя прочитать уверенно, поставь [НЕРАЗБОРЧИВО].
Не используй Markdown или пояснения вне JSON.
Верни JSON-объект с полями page_num и tasks — списком всех заданий страницы.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "page_num": {"type": "integer"},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_num": {"type": "integer"},
                    "condition": {"type": "string"},
                },
                "required": ["task_num", "condition"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["page_num", "tasks"],
    "additionalProperties": False,
}


def validate_conditions(items, *, require_all=True):
    """Проверить структуру ответа, но не математическую точность формул."""
    if not isinstance(items, list):
        raise ValueError("Gemini должен вернуть список заданий")

    conditions = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"task_num", "condition"}:
            raise ValueError("В записи задания нужны только task_num и condition")
        number = item.get("task_num")
        condition = item.get("condition")
        if type(number) is not int or number not in TASK_NUMBERS:
            raise ValueError(f"Некорректный номер задания: {number}")
        if number in conditions:
            raise ValueError(f"Повтор задания: {number}")
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(f"Пустое условие задания {number}")
        if "[НЕРАЗБОРЧИВО]" in condition:
            raise ValueError(f"Задание {number}: нужна проверка по PDF")
        if "\ufffd" in condition or any(
            unicodedata.category(char) == "Co" for char in condition
        ):
            raise ValueError(f"Задание {number}: повреждённые символы")
        conditions[number] = condition.strip()

    if require_all and sorted(conditions) != TASK_NUMBERS:
        raise ValueError("Gemini вернул не все задания 1–19")
    return conditions


def get_task_positions(page):
    """Найти номера в начале строк; используется также для привязки рисунков."""
    positions = []
    for x0, y0, x1, y1, word, block, line, word_num in page.get_text("words"):
        match = TASK_RE.fullmatch(word) if word_num == 0 else None
        if match is not None:
            positions.append({"task_num": int(match.group(1)), "x": x0, "y": y0})
    return positions


def render_page_images(pdf_path, image_dir):
    """Сохранить полные страницы и найти ожидаемые номера по текстовому слою."""
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    with pymupdf.open(pdf_path) as document:
        all_numbers = []
        for page_num, page in enumerate(document, start=1):
            numbers = sorted(item["task_num"] for item in get_task_positions(page))
            pages.append({"page_num": page_num, "task_numbers": numbers})
            all_numbers.extend(numbers)
        if sorted(all_numbers) != TASK_NUMBERS:
            missing = sorted(set(TASK_NUMBERS) - set(all_numbers))
            raise ValueError(
                "В PDF нужны задания 1–19 без повторов. "
                f"Не найдены: {missing}; получены номера: {all_numbers}"
            )
        for page in pages:
            image_path = image_dir / f"page_{page['page_num']:02d}.png"
            pixmap = document[page["page_num"] - 1].get_pixmap(
                dpi=DPI, colorspace=pymupdf.csRGB, alpha=False,
            )
            image = pixmap.tobytes("png")
            image_path.write_bytes(image)
            page.update(
                image_path=image_path,
                image_sha256=hashlib.sha256(image).hexdigest(),
            )
    return pages


def validate_page_response(data, page_num, expected_numbers):
    """Проверить один полный ответ, не исправляя его новым запросом."""
    if not isinstance(data, dict) or set(data) != {"page_num", "tasks"}:
        raise ValueError("Нужен объект с полями page_num и tasks")
    if type(data["page_num"]) is not int or data["page_num"] != page_num:
        raise ValueError(f"Ожидался номер страницы {page_num}")
    conditions = validate_conditions(data["tasks"], require_all=False)
    if sorted(conditions) != sorted(expected_numbers):
        raise ValueError(
            f"Страница {page_num}: ожидались задания {sorted(expected_numbers)}, "
            f"получены {sorted(conditions)}"
        )
    return conditions


def atomic_write_json(path, data):
    """Старый файл сохраняется при ошибке записи или атомарной замены."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as target:
            pending = Path(target.name)
            json.dump(data, target, ensure_ascii=False, indent=2)
        pending.replace(path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def write_cache(cache_path, cached, conditions):
    """Сохранить общий результат в прежнем формате списка tasks."""
    data = {
        **cached,
        "tasks": [
            {"task_num": number, "condition": conditions[number]}
            for number in sorted(conditions)
        ],
    }
    atomic_write_json(cache_path, data)


def legacy_cache_identity(pdf_path):
    """Точная конфигурация прежнего OCR: только полный кэш пригоден к чтению."""
    return {
        "version": CACHE_VERSION,
        "method": METHOD,
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "model": MODEL,
        "prompt": PROMPT,
        "dpi": DPI,
    }


def image_parameters():
    return {"dpi": DPI, "colorspace": "RGB", "alpha": False}


def aggregate_cache_identity(pdf_path):
    return {
        "version": AGGREGATE_CACHE_VERSION,
        "method": PAGE_METHOD,
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "model": MODEL,
        "prompt": PAGE_PROMPT,
        "image_parameters": image_parameters(),
        "schema": SCHEMA,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "timeout_ms": REQUEST_TIMEOUT_MS,
        "http_attempts": 1,
    }


def page_cache_identity(page):
    # Здесь нет хеша всего PDF: изменение другой страницы не отменяет этот кэш.
    return {
        "version": PAGE_CACHE_VERSION,
        "method": PAGE_METHOD,
        "page_num": page["page_num"],
        "image_sha256": page["image_sha256"],
        "task_numbers": page["task_numbers"],
        "model": MODEL,
        "prompt": PAGE_PROMPT,
        "image_parameters": image_parameters(),
        "schema": SCHEMA,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "timeout_ms": REQUEST_TIMEOUT_MS,
        "http_attempts": 1,
    }


def read_complete_cache(cache_path, identities):
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Нужен JSON-объект")
        if not any(
            all(data.get(key) == value for key, value in identity.items())
            for identity in identities
        ):
            raise ValueError("изменились исходник или настройки OCR")
        return validate_conditions(data["tasks"])
    except (ValueError, TypeError, KeyError) as error:
        print(f"Полный OCR-кэш не подходит: {cache_path}: {error}", flush=True)
        return None


def recognize_conditions(pdf_path):
    """Один вызов на отсутствующую страницу; пригодный полный кэш — ноль."""
    pdf_path = Path(pdf_path)
    cache_path = pdf_path.with_name(f"{pdf_path.stem}_conditions_gemini.json")
    identity = aggregate_cache_identity(pdf_path)
    conditions = read_complete_cache(cache_path, (identity, legacy_cache_identity(pdf_path)))
    if conditions is not None:
        print(
            f"Условия прочитаны из OCR-кэша: {cache_path}. Запросов OCR: 0.",
            flush=True,
        )
        return conditions

    ocr_dir = pdf_path.parent / f"{pdf_path.stem}_ocr"
    pages = render_page_images(pdf_path, ocr_dir)
    missing = []
    conditions = {}
    for page in pages:
        page["cache_path"] = ocr_dir / f"page_{page['page_num']:02d}.json"
        page["identity"] = page_cache_identity(page)
        if page["cache_path"].is_file():
            try:
                cached = json.loads(page["cache_path"].read_text(encoding="utf-8"))
                if not isinstance(cached, dict) or cached.get("identity") != page["identity"]:
                    raise ValueError("изменились страница или настройки OCR")
                result = validate_page_response(
                    cached["response"], page["page_num"], page["task_numbers"],
                )
            except (ValueError, TypeError, KeyError) as error:
                print(f"Кэш страницы {page['page_num']} не подходит: {error}", flush=True)
            else:
                conditions.update(result)
                print(f"Страница {page['page_num']} прочитана из кэша.", flush=True)
                continue
        missing.append(page)

    requests_used = 0
    if missing:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Для недостающих страниц OCR не задана переменная GEMINI_API_KEY")
        with genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ) as client:
            for page in missing:
                page_num = page["page_num"]
                requests_used += 1
                print(
                    f"Страница {page_num}/{len(pages)}: OCR через {MODEL}. "
                    f"Запрос OCR {requests_used}/{len(missing)}.",
                    flush=True,
                )
                try:
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=[
                            f"Номер страницы: {page_num}",
                            types.Part.from_bytes(
                                data=page["image_path"].read_bytes(), mime_type="image/png",
                            ),
                            PAGE_PROMPT,
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_json_schema=SCHEMA,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                        ),
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"Страница {page_num}: ошибка API или сети при OCR: {error}. "
                        f"Запросов OCR: {requests_used}. Автоматических повторов нет; "
                        "проверенные кэши страниц и прежняя выгрузка сохранены."
                    ) from error
                try:
                    if not response.text:
                        raise ValueError("Gemini вернул ответ без текста")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    raw_path = ocr_dir / f"page_{page_num:02d}_response_{timestamp}.txt"
                    raw_path.write_text(response.text, encoding="utf-8")
                    if any(
                        getattr(getattr(candidate, "finish_reason", None), "value", None)
                        == "MAX_TOKENS"
                        or getattr(candidate, "finish_reason", None) == "MAX_TOKENS"
                        for candidate in (getattr(response, "candidates", None) or ())
                    ):
                        raise ValueError("Gemini оборвал ответ по ограничению длины MAX_TOKENS")
                    data = json.loads(response.text)
                    result = validate_page_response(data, page_num, page["task_numbers"])
                except (ValueError, TypeError) as error:
                    raise ValueError(
                        f"Страница {page_num}: некорректный ответ OCR: {error}. "
                        f"Запросов OCR: {requests_used}. Дополнительный запрос не выполняется."
                    ) from error
                atomic_write_json(page["cache_path"], {
                    "identity": page["identity"], "response": data,
                    "raw_response": raw_path.name,
                })
                conditions.update(result)
                print(f"Страница {page_num} проверена и сохранена.", flush=True)

    # Состав каждого ответа уже совпал с номерами своей страницы. Общая
    # проверка дополнительно требует ровно 19 разных непустых условий.
    validate_conditions([
        {"task_num": number, "condition": condition}
        for number, condition in conditions.items()
    ])
    write_cache(cache_path, {
        **identity,
        "page_caches": [str(page["cache_path"].relative_to(pdf_path.parent)) for page in pages],
    }, conditions)
    print(
        f"OCR завершён: 19/19, страниц: {len(pages)}. "
        f"Запросов OCR: {requests_used}. JSON: {cache_path}",
        flush=True,
    )
    return conditions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR условий из локального PDF")
    parser.add_argument("--pdf", type=Path, required=True, help="Путь к скачанному PDF")
    args = parser.parse_args()
    recognize_conditions(args.pdf)
