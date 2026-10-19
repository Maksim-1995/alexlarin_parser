"""Повторно распознаёт выбранные задания по крупным фрагментам PDF.

Запуск из корня проекта: python -u refine_conditions.py 6 13 18
Фрагменты и ответ Gemini сохраняются в input/<вариант>_ocr_review/.
Проверенный по структуре ответ обновляет существующий кэш условий.
Для проверки математического содержания нужна сверка с PDF.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pymupdf
from google import genai
from google.genai import types

from parsers import TASK_RE
from recognizer import MODEL, validate_conditions


PROMPT = r"""
Перепиши полные условия заданий с приложенных изображений.
Перед каждым изображением указан номер задания.
Сохрани исходную математическую запись, а не равносильное преобразование.
Не решай задачи, не добавляй ответы или пояснения.
Сохрани все подпункты, ограничения и слова условия.
Особенно внимательно прочитай дробные черты, корни, основания логарифмов,
показатели степеней, скобки и расположение выражений в строках системы.
Убедись, что каждый корень относится именно к тому выражению, над которым
проходит его черта. Не переноси выражения из одной строки системы в другую.
Запиши формулы в LaTeX. Номер задания запиши только в task_num.
Убери бланки «Ответ: ____», колонтитулы и инструкции экзамена.
Если фрагмент нельзя прочитать уверенно, поставь [НЕРАЗБОРЧИВО].
"""


def render_task_images(pdf_path, task_nums, review_dir):
    """Сохранить выбранные области двухколоночного PDF с разрешением 300 dpi."""
    found = {}

    with pymupdf.open(pdf_path) as document:
        for page in document:
            positions = []
            for word_data in page.get_text("words"):
                x0, y0, x1, y1, word, block_num, line_num, word_num = word_data
                match = TASK_RE.fullmatch(word) if word_num == 0 else None
                if match is not None:
                    positions.append((int(match.group(1)), x0, y0))

            middle_x = page.rect.width / 2
            for left, right in ((0, middle_x), (middle_x, page.rect.width)):
                column = sorted(
                    (item for item in positions if left <= item[1] < right),
                    key=lambda item: item[2],
                )
                for index, (number, x0, y0) in enumerate(column):
                    if number not in task_nums:
                        continue
                    if number in found:
                        raise ValueError(f"Повторная область задания {number}")

                    top = max(0, y0 - 12)
                    bottom = (
                        column[index + 1][2] - 12
                        if index + 1 < len(column)
                        else page.rect.height - 25
                    )
                    area = pymupdf.Rect(left, top, right, bottom)
                    image_path = review_dir / f"task_{number:02d}.png"
                    page.get_pixmap(clip=area, dpi=300).save(image_path)
                    found[number] = image_path

    if set(found) != set(task_nums):
        raise ValueError(f"Не найдены области заданий: {sorted(set(task_nums) - set(found))}")
    return found


def refine_conditions(pdf_path, task_nums):
    """Обновить только выбранные условия; сохранить прежний кэш и ответ API."""
    task_nums = sorted(task_nums)
    if not task_nums or len(task_nums) != len(set(task_nums)):
        raise ValueError("Укажи неповторяющиеся номера заданий")
    if any(number not in range(1, 20) for number in task_nums):
        raise ValueError("Допустимы номера от 1 до 19")

    pdf_path = Path(pdf_path)
    cache_path = pdf_path.with_name(f"{pdf_path.stem}_conditions_gemini.json")
    original_text = cache_path.read_text(encoding="utf-8")
    cached = json.loads(original_text)
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    if cached.get("pdf_sha256") != pdf_hash:
        raise ValueError("Кэш относится к другому PDF. Сначала выполни основное распознавание")
    if cached.get("model") != MODEL:
        raise ValueError("Модель кэша отличается от модели в recognizer.py")
    validate_conditions(cached["tasks"])

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Не задана переменная GEMINI_API_KEY")

    review_dir = pdf_path.parent / f"{pdf_path.stem}_ocr_review"
    review_dir.mkdir(exist_ok=True)
    images = render_task_images(pdf_path, task_nums, review_dir)

    contents = []
    for number in task_nums:
        contents.extend([
            f"Задание {number}",
            types.Part.from_bytes(
                data=images[number].read_bytes(), mime_type="image/png"
            ),
        ])
    contents.append(PROMPT)

    schema = {
        "type": "array",
        "minItems": len(task_nums),
        "maxItems": len(task_nums),
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

    print(f"Повторно распознаём задания {task_nums}. Тайм-аут — 120 секунд...", flush=True)
    with genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema,
                max_output_tokens=8192,
            ),
        )

    if not response.text:
        raise ValueError("Gemini вернул ответ без текста")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_path = review_dir / f"gemini_response_{timestamp}.txt"
    raw_path.write_text(response.text, encoding="utf-8")
    items = json.loads(response.text)
    if not isinstance(items, list):
        raise ValueError("Ожидался список заданий")

    replacements = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Некорректная запись задания")
        number = item.get("task_num")
        if type(number) is not int or number not in task_nums or number in replacements:
            raise ValueError(f"Некорректный или повторный номер задания: {number}")
        replacements[number] = item
    if sorted(replacements) != task_nums:
        raise ValueError("Gemini вернул не все выбранные задания")

    merged = [replacements.get(item["task_num"], item) for item in cached["tasks"]]
    conditions = validate_conditions(merged)

    backup_path = review_dir / f"conditions_before_{timestamp}.json"
    backup_path.write_text(original_text, encoding="utf-8")
    cached["tasks"] = merged
    cached.setdefault("refinements", []).append({
        "task_nums": task_nums,
        "method": "gemini_image_transcription",
        "model": MODEL,
        "pdf_sha256": pdf_hash,
        "dpi": 300,
        "prompt": PROMPT,
        "raw_response": raw_path.name,
    })
    pending_path = cache_path.with_suffix(".tmp")
    pending_path.write_text(
        json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pending_path.replace(cache_path)

    print(f"Кэш обновлён: {cache_path}", flush=True)
    print(f"Прежний кэш сохранён: {backup_path}", flush=True)
    for number in task_nums:
        print(f"\nЗадание {number}:\n{conditions[number]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Уточнение условий по крупным фрагментам PDF")
    parser.add_argument("task_nums", nargs="+", type=int)
    parser.add_argument(
        "--pdf", type=Path,
        default=Path(__file__).resolve().parent / "input" / "trvar542.pdf",
    )
    args = parser.parse_args()
    refine_conditions(args.pdf, args.task_nums)
