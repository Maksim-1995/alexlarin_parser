import fitz

import re

from pathlib import Path

import requests

from urllib.parse import urljoin


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

with fitz.open(pdf_path) as document:
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
