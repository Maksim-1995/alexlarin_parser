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
