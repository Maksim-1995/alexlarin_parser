"""Контрольная проверка OCR второй страницы варианта 542."""

import base64
import json
import os
import time

import pymupdf
import requests


api_key = os.environ.get("DEEPSEEK_API_KEY")

if not api_key:
    raise SystemExit("Не задан DEEPSEEK_API_KEY.")

model = "deepseek-flash"
image_parts = []

with pymupdf.open("input/trvar542.pdf") as document:
    page = document[1]
    rect = page.rect
    middle = (rect.x0 + rect.x1) / 2

    columns = [
        pymupdf.Rect(rect.x0, rect.y0, middle + 5, rect.y1),
        pymupdf.Rect(middle - 5, rect.y0, rect.x1, rect.y1),
    ]

    for column in columns:
        image = page.get_pixmap(
            dpi=300,
            clip=column,
            alpha=False,
        ).tobytes("png")

        encoded = base64.b64encode(image).decode()

        image_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded}",
                "detail": "high",
            },
        })

prompt = (
    "Перед тобой два изображения: левая и правая колонки "
    "одной страницы экзаменационного варианта. "
    "Перепиши полностью и дословно условия всех заданий 5–12. "
    "Сохрани русский текст, числа, знаки, единицы измерения "
    "и указанную точность округления. "
    "Формулы запиши в LaTeX внутри $...$. "
    r"Знак градуса записывай как ^{\circ}, различай его и степень ноль. "
    "Сохраняй пробелы между словами. "
    "Не решай задания, не добавляй пояснений. "
    "Не включай строки для записи ответа, колонтитулы и инструкции экзамена. "
    "Неразборчивые фрагменты обозначай [неразборчиво], не угадывай. "
    "Верни только JSON-объект с полем tasks. "
    "Это массив из восьми объектов с номерами от 5 до 12 по порядку. "
    "Каждый объект содержит целочисленное поле task_num "
    "и строковое поле condition с полным условием. "
    "Пример структуры одного задания: "
    '{"tasks": [{"task_num": 5, "condition": "Полное условие"}]}. '
    "В твоём ответе должны присутствовать все восемь заданий. "
    "Корректно экранируй обратные слеши LaTeX внутри JSON."
)

print("Провайдер: DeepSeek напрямую")
print("Модель:", model)
print("Страница: 2. Изображений: 2. Запросов: 1.", flush=True)

started = time.monotonic()

try:
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        json={
            "model": model,
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_parts,
                ],
            }],
            "max_tokens": 12000,
            "stream": False,
        },
        timeout=(15, 120),
    )
except requests.RequestException as error:
    print("Секунд:", round(time.monotonic() - started, 1))
    raise SystemExit(f"{type(error).__name__}: {error}")

print("HTTP:", response.status_code)
print("Секунд:", round(time.monotonic() - started, 1))

try:
    body = response.json()
except ValueError:
    raise SystemExit(f"Ответ API вне JSON: {response.text[:2000]}")

if not response.ok or not body.get("choices"):
    raise SystemExit(json.dumps(body, ensure_ascii=False, indent=2))

choice = body["choices"][0]
content = choice.get("message", {}).get("content")

print("Причина завершения:", choice.get("finish_reason"))
print(
    "Расход токенов:",
    json.dumps(body.get("usage"), ensure_ascii=False),
)

if not isinstance(content, str) or not content.strip():
    raise SystemExit("Модель вернула пустой текст распознавания.")

print("\nПолный ответ модели:")
print(content)

if choice.get("finish_reason") == "length":
    raise SystemExit("Ответ оборван по лимиту токенов.")

try:
    result = json.loads(content)
except ValueError as error:
    raise SystemExit(f"Некорректный JSON модели: {error}")

if not isinstance(result, dict) or not isinstance(result.get("tasks"), list):
    raise SystemExit("Ожидался JSON-объект с массивом tasks.")

tasks = result["tasks"]

if not all(isinstance(task, dict) for task in tasks):
    raise SystemExit("Задания должны быть JSON-объектами.")

numbers = [task.get("task_num") for task in tasks]
print("\nПолучены номера:", numbers)

if numbers != list(range(5, 13)):
    raise SystemExit("Ожидались задания 5–12, каждое ровно один раз.")

if not all(
    isinstance(task.get("condition"), str) and task["condition"].strip()
    for task in tasks
):
    raise SystemExit("Обнаружено отсутствующее или пустое условие.")

print("Структура корректна: восемь непустых условий.")
print("Точность текста и формул нужно проверить по PDF.")