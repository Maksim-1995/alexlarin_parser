"""Распознавание условий из исходного PDF через Gemini."""

import hashlib
import json
import os
import unicodedata
from pathlib import Path

from google import genai
from google.genai import types


MODEL = "gemini-3.5-flash-lite"

PROMPT = r"""
Перепиши полные условия математических заданий 1–19 из этого PDF.
Используй визуальный вид страниц, особенно при чтении формул.
На страницах две колонки: читай каждую колонку сверху вниз.
Сохрани исходные формулировки и все подпункты А), Б), В).
Номер задания укажи в task_num, не повторяй его в condition.
Не включай заголовки страниц, инструкции экзамена и строки «Ответ: ___».
Не решай задания, не добавляй ответы и пояснения от себя.
Формулы записывай в LaTeX: \frac{a}{b}, \sqrt{x}, x^{2}, \vec{a}.
Не используй блоки Markdown. Не заменяй формулы описанием словами.
Не теряй скобки, знаки, показатели степеней и основания логарифмов.
Если фрагмент нельзя прочитать уверенно, поставь [НЕРАЗБОРЧИВО].
"""

SCHEMA = {
    "type": "array",
    "minItems": 19,
    "maxItems": 19,
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


def validate_conditions(items):
    """Проверить номера, непустой текст и повреждённые символы."""
    if not isinstance(items, list):
        raise ValueError("Gemini должен вернуть список заданий")

    conditions = {}

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Некорректная запись задания")

        number = item.get("task_num")
        condition = item.get("condition")

        if type(number) is not int or number not in range(1, 20):
            raise ValueError(f"Некорректный номер задания: {number}")

        if number in conditions:
            raise ValueError(f"Повтор задания: {number}")

        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(f"Пустое условие задания {number}")

        if "[НЕРАЗБОРЧИВО]" in condition:
            raise ValueError(f"Задание {number}: нужна проверка по PDF")

        if "\ufffd" in condition or any(
            unicodedata.category(char) == "Co"
            for char in condition
        ):
            raise ValueError(
                f"Задание {number}: обнаружены повреждённые символы"
            )

        conditions[number] = condition.strip()

    if sorted(conditions) != list(range(1, 20)):
        raise ValueError("Gemini вернул не все задания 1–19")

    return conditions


def recognize_conditions(pdf_path):
    """Распознать условия или загрузить результат для того же PDF."""
    pdf_path = Path(pdf_path)
    pdf_data = pdf_path.read_bytes()
    pdf_hash = hashlib.sha256(pdf_data).hexdigest()

    cache_path = pdf_path.with_name(
        f"{pdf_path.stem}_conditions_gemini.json"
    )

    if cache_path.exists():
        cached = json.loads(
            cache_path.read_text(encoding="utf-8")
        )

        if (
            cached.get("pdf_sha256") == pdf_hash
            and cached.get("model") == MODEL
            and cached.get("prompt") == PROMPT
        ):
            conditions = validate_conditions(cached["tasks"])
            print(
                f"Условия прочитаны из кэша: {cache_path}",
                flush=True,
            )
            return conditions

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Не задана переменная GEMINI_API_KEY")

    print(
        f"Распознаём PDF через {MODEL}. "
        "Тайм-аут — 120 секунд...",
        flush=True,
    )

    with genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(
                    data=pdf_data,
                    mime_type="application/pdf",
                ),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=SCHEMA,
                max_output_tokens=16_384,
            ),
        )

    if not response.text:
        raise ValueError("Gemini вернул ответ без текста")

    # Исходный ответ сохраняем для разбора возможных ошибок.
    raw_path = pdf_path.with_name(
        f"{pdf_path.stem}_gemini_raw.txt"
    )
    raw_path.write_text(response.text, encoding="utf-8")

    items = json.loads(response.text)
    conditions = validate_conditions(items)

    cached = {
        "pdf_sha256": pdf_hash,
        "model": MODEL,
        "prompt": PROMPT,
        "tasks": items,
    }

    cache_path.write_text(
        json.dumps(cached, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Распознано условий: {len(conditions)}. "
        f"JSON: {cache_path}",
        flush=True,
    )

    return conditions
