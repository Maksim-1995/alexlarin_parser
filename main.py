import jsbeautifier

from jsbeautifier.unpackers import packer
import pymupdf

import re

from pathlib import Path

import requests

from urllib.parse import unquote, urljoin


def clean_condition(text):
    # Удаляем пустое поле для записи ответа.
    text = re.sub(
        r"Ответ:[ \t]*_+\.?",
        "",
        text,
    )

    # Удаляем строки с колонтитулами.
    text = re.sub(
        r"^.*(?:Единый государственный экзамен,|alexlarin\.net|Разрешается свободное копирование).*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Отсекаем инструкции, попавшие после условия.
    text = re.split(
        r"^[ \t]*(?:Не забудьте перенести|Часть[ \t]+2\b|Проверьте, чтобы)",
        text,
        maxsplit=1,
        flags=re.MULTILINE,
    )[0]

    # Удаляем пробелы по краям каждой строки.
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    # Три и более переноса заменяем двумя.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


PAGE_URL = "https://alexlarin.net/ege/2026/trvar542.html"

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"


INPUT_DIR.mkdir(exist_ok=True)

response = requests.get(PAGE_URL, timeout=30)
response.raise_for_status()

response.encoding = "windows-1251"
html = response.text

html_path = INPUT_DIR / "trvar542.html"
html_path.write_text(html, encoding="utf-8")

print(f"HTML сохранён: {html_path}")

pdf_match = re.search(
    r'href\s*=\s*["\']([^"\']+\.pdf)["\']',
    html,
    flags=re.IGNORECASE,
)

if pdf_match is None:
    raise ValueError("На странице не найдена ссылка на PDF")

pdf_link = pdf_match.group(1)
pdf_url = urljoin(PAGE_URL, pdf_link)

print(f"Найдена ссылка на PDF: {pdf_url}")

pdf_response = requests.get(pdf_url, timeout=30)
pdf_response.raise_for_status()

pdf_data = pdf_response.content

if not pdf_data.startswith(b"%PDF-"):
    raise ValueError("Скачанный файл не похож на PDF")

pdf_name = Path(pdf_link).name
pdf_path = INPUT_DIR / pdf_name
pdf_path.write_bytes(pdf_data)

print(f"PDF сохранён: {pdf_path}")


OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

page_texts = []

with pymupdf.open(pdf_path) as document:
    print(f"Количество страниц: {len(document)}")

    for page in document:
        page_text = page.get_text()
        page_texts.append(page_text)

full_text = "\n".join(page_texts)

if not full_text.strip():
    raise ValueError("Не удалось извлечь текст из PDF")

text_path = OUTPUT_DIR / "raw_text.txt"
text_path.write_text(full_text, encoding="utf-8")

print(f"Извлечено символов: {len(full_text)}")
print(f"Текст сохранён: {text_path}")


TASK_RE = re.compile(
    r"^[ \t]*(1[0-9]|[1-9])\.(?=[ \t]|$)",
    flags=re.MULTILINE,
)

task_matches = list(TASK_RE.finditer(full_text))

task_numbers = [
    int(match.group(1))
    for match in task_matches
]

print(f"Найдены номера: {task_numbers}")

if task_numbers != list(range(1, 20)):
    raise ValueError("Ожидались задания с номерами от 1 до 19 по порядку")

tasks = []

for index, match in enumerate(task_matches):
    condition_start = match.end()

    if index + 1 < len(task_matches):
        condition_end = task_matches[index + 1].start()
    else:
        condition_end = len(full_text)

    condition = full_text[condition_start:condition_end]
    condition = clean_condition(condition)

    if not condition:
        raise ValueError(
            f"Пустое условие задания {match.group(1)}"
        )

    tasks.append({
        "task_num": int(match.group(1)),
        "condition": condition,
    })

print(f"Выделено заданий: {len(tasks)}")

for task_index in (0, 11, 18):
    task = tasks[task_index]

    print(f"\nЗадание {task['task_num']}:")
    print(task["condition"])


IMAGES_DIR = OUTPUT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

saved_xrefs = set()

with pymupdf.open(pdf_path) as document:
    for page_number, page in enumerate(document, start=1):
        images = page.get_images()

        for image in images:
            xref = image[0]

            if xref in saved_xrefs:
                continue

            image_data = document.extract_image(xref)

            extension = image_data["ext"]
            image_name = f"image_{xref}.{extension}"
            image_path = IMAGES_DIR / image_name

            image_path.write_bytes(image_data["image"])
            saved_xrefs.add(xref)

            print(f"Страница {page_number}: сохранён {image_name}")

print(f"Всего сохранено изображений: {len(saved_xrefs)}")


script_sources = re.findall(
    r'<script\b[^>]*\bsrc\s*=\s*["\']([^"\']+)["\']',
    html,
    flags=re.IGNORECASE,
)

expected_js_name = f"{pdf_path.stem}.js"
js_url = None

for source in script_sources:
    decoded_source = unquote(source)

    if Path(decoded_source).name == expected_js_name:
        js_url = urljoin(PAGE_URL, decoded_source)
        break

if js_url is None:
    raise ValueError("Не найден JavaScript-файл этого варианта")

print(f"Найдена ссылка на JavaScript: {js_url}")

js_response = requests.get(js_url, timeout=30)
js_response.raise_for_status()

js_data = js_response.content
js_path = INPUT_DIR / expected_js_name
js_path.write_bytes(js_data)

print(f"JavaScript сохранён: {js_path}")

js_text = js_data.decode("utf-16")

print("Начало JavaScript:")
print(js_text[:200])

if packer.detect(js_text):
    unpacked_js = packer.unpack(js_text)
else:
    unpacked_js = js_text

readable_js = jsbeautifier.beautify(unpacked_js)

unpacked_path = OUTPUT_DIR / f"{pdf_path.stem}_unpacked.js"
unpacked_path.write_text(readable_js, encoding="utf-8")

print(f"Распакованный JavaScript сохранён: {unpacked_path}")

ANSWER_RE = re.compile(
    r'getElementById\(\s*["\']T(\d+)["\']\s*\)\.value\s*=\s*["\']([^"\']+)["\']'
)

answer_matches = ANSWER_RE.findall(readable_js)

answer_numbers = sorted(
    int(number)
    for number, value in answer_matches
)

if answer_numbers != list(range(1, 13)):
    raise ValueError("Ожидались ответы на задания 1–12 без повторов")

answers = {}

for number, value in answer_matches:
    answers[int(number)] = value

print(f"Извлечено ответов: {len(answers)}")

for number, value in sorted(answers.items()):
    print(f"{number}: {value}")

for task in tasks:
    task["answer"] = answers.get(task["task_num"], "")
