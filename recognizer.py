"""OCR условий по крупным изображениям заданий через Gemini.

Поддерживается двухколоночная вёрстка с номерами 1.–19. в текстовом слое.
Номера нужны только для поиска областей; условия модель читает с изображений.
Служебные PNG, ответы API и резервные копии кэша сохраняются в input/.

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

import pymupdf
from google import genai
from google.genai import types

from parsers import TASK_RE


MODEL = "gemini-3.5-flash-lite"
METHOD = "gemini_image_ocr"
DPI = 300
CACHE_VERSION = 2
TASK_NUMBERS = list(range(1, 20))

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

SCHEMA = {
    "type": "array",
    "minItems": 1,
    "maxItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "task_num": {"type": "integer"},
            "condition": {"type": "string"},
        },
        "required": ["task_num", "condition"],
        "additionalProperties": False,
    },
}


def validate_conditions(items, *, require_all=True):
    """Проверить структуру ответа, но не математическую точность формул."""
    if not isinstance(items, list):
        raise ValueError("Gemini должен вернуть список заданий")

    conditions = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Некорректная запись задания")
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


def render_task_images(pdf_path, image_dir):
    """Вырезать области от номера задания до следующего номера в колонке."""
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    areas = {}
    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            positions = get_task_positions(page)
            middle_x = page.rect.width / 2
            for left, right in ((0, middle_x), (middle_x, page.rect.width)):
                column = sorted(
                    (item for item in positions if left <= item["x"] < right),
                    key=lambda item: item["y"],
                )
                for index, position in enumerate(column):
                    number = position["task_num"]
                    if number in areas:
                        raise ValueError(f"Повторная область задания {number}")
                    top = max(0, position["y"] - 12)
                    bottom = (
                        column[index + 1]["y"] - 12
                        if index + 1 < len(column)
                        else page.rect.height - 25
                    )
                    if bottom <= top:
                        raise ValueError(f"Не удалось определить область задания {number}")
                    areas[number] = (page_index, pymupdf.Rect(left, top, right, bottom))

        if sorted(areas) != TASK_NUMBERS:
            missing = sorted(set(TASK_NUMBERS) - set(areas))
            raise ValueError(
                f"Не найдены области заданий: {missing}. "
                "Нужен PDF с двумя колонками и номерами в текстовом слое"
            )

        images = {}
        for number in TASK_NUMBERS:
            page_index, area = areas[number]
            image_path = image_dir / f"task_{number:02d}.png"
            document[page_index].get_pixmap(clip=area, dpi=DPI).save(image_path)
            images[number] = image_path
    return images


def write_cache(cache_path, cached, conditions):
    """Заменить кэш целиком после успешной записи временного файла."""
    cached["tasks"] = [
        {"task_num": number, "condition": conditions[number]}
        for number in sorted(conditions)
    ]
    pending_path = cache_path.with_suffix(".tmp")
    pending_path.write_text(
        json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pending_path.replace(cache_path)


def recognize_conditions(pdf_path):
    """Распознавать по одному заданию, продолжая незавершённую обработку."""
    pdf_path = Path(pdf_path)
    identity = {
        "version": CACHE_VERSION,
        "method": METHOD,
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "model": MODEL,
        "prompt": PROMPT,
        "dpi": DPI,
    }
    cache_path = pdf_path.with_name(f"{pdf_path.stem}_conditions_gemini.json")
    cached = dict(identity)
    conditions = {}
    previous_text = None

    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        try:
            previous = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"Повреждён JSON-кэш: {cache_path}") from error
        if not isinstance(previous, dict):
            raise ValueError(f"Некорректный JSON-кэш: {cache_path}")
        if all(previous.get(key) == value for key, value in identity.items()):
            conditions = validate_conditions(previous["tasks"], require_all=False)
            cached = previous
            if len(conditions) == len(TASK_NUMBERS):
                print(f"Условия прочитаны из OCR-кэша: {cache_path}", flush=True)
                return conditions
            print(f"В OCR-кэше готово {len(conditions)}/19 заданий. Продолжаем.", flush=True)
        else:
            previous_text = text

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Не задана переменная GEMINI_API_KEY")

    ocr_dir = pdf_path.parent / f"{pdf_path.stem}_ocr"
    images = render_task_images(pdf_path, ocr_dir)
    print(f"Изображения для OCR: {ocr_dir}; разрешение: {DPI} dpi.", flush=True)

    if previous_text is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = ocr_dir / f"conditions_before_{timestamp}.json"
        backup_path.write_text(previous_text, encoding="utf-8")
        print(f"Прежний кэш сохранён: {backup_path}", flush=True)

    with genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        for number in TASK_NUMBERS:
            if number in conditions:
                continue
            print(f"Задание {number}/19: OCR через {MODEL}. Тайм-аут — 120 секунд...", flush=True)
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    f"Номер задания: {number}",
                    types.Part.from_bytes(
                        data=images[number].read_bytes(), mime_type="image/png"
                    ),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=SCHEMA,
                    max_output_tokens=4096,
                ),
            )
            if not response.text:
                raise ValueError(f"Задание {number}: Gemini вернул ответ без текста")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            raw_path = ocr_dir / f"task_{number:02d}_response_{timestamp}.txt"
            raw_path.write_text(response.text, encoding="utf-8")
            result = validate_conditions(json.loads(response.text), require_all=False)
            if set(result) != {number}:
                raise ValueError(f"Ожидалось только задание {number}, получено: {sorted(result)}")

            conditions.update(result)
            cached.setdefault("raw_responses", {})[str(number)] = str(
                raw_path.relative_to(pdf_path.parent)
            )
            write_cache(cache_path, cached, conditions)
            print(f"Задание {number} сохранено. Готово: {len(conditions)}/19.", flush=True)

    validate_conditions(cached["tasks"])
    print(f"OCR завершён: 19/19. JSON: {cache_path}", flush=True)
    return conditions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR условий из локального PDF")
    parser.add_argument("--pdf", type=Path, required=True, help="Путь к скачанному PDF")
    args = parser.parse_args()
    recognize_conditions(args.pdf)
