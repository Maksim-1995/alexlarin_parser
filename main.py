"""Собирает вариант alexlarin.net в Excel и сохраняет рисунки заданий.

Исходники и промежуточные файлы находятся в input/.
В output/<variant_id>/ сохраняются только tasks.xlsx и рисунки заданий.
ZIP создаётся рядом с main.py.

PyMuPDF находит номера заданий и встроенные рисунки.
Gemini распознаёт условия по изображениям заданий с разрешением 300 dpi.
Перед записью в Excel обозначения оформляются в LaTeX регулярными выражениями.
"""

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import jsbeautifier
import pymupdf
from jsbeautifier.unpackers import packer

from classification import classify_tasks
from downloader import download_bytes
from exporter import create_zip, write_excel
from export_metadata import prepare_about_metadata
from latex_format import normalize_condition
from parsers import parse_js_answers, parse_tasks
from recognizer import DPI, METHOD, MODEL, get_task_positions, recognize_conditions
from task_html import add_subtask_breaks


DEFAULT_PAGE_URL = "https://alexlarin.net/ege/2026/trvar542.html"
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"


def parse_metadata(text, source_url, export_date):
    """Извлечь класс, год экзамена и название варианта из текста PDF."""
    class_match = re.search(
        r"\b(\d{1,2})\s*класс\b", text, flags=re.IGNORECASE
    )
    year_match = re.search(
        r"экзамен,\s*(\d{4})\s*г\b", text, flags=re.IGNORECASE
    )
    variant_match = re.search(
        r"Тренировочный\s+вариант\s*№\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if not (class_match and year_match and variant_match):
        raise ValueError("Не удалось найти класс, год или номер варианта")

    return {
        "class": int(class_match.group(1)),
        "year": int(year_match.group(1)),
        "date": export_date,
        "variant_name": (
            f"Тренировочный вариант № {variant_match.group(1)}"
        ),
        "source_url": source_url,
        "date_kind": "Дата формирования выгрузки",
    }


def find_image_task(positions, center_x, center_y, middle_x):
    """Найти ближайшее задание выше рисунка в той же колонке."""
    candidates = []
    image_is_right = center_x >= middle_x

    for position in positions:
        task_is_right = position["x"] >= middle_x
        same_column = task_is_right == image_is_right
        starts_above = position["y"] <= center_y

        if same_column and starts_above:
            candidates.append(position)

    if not candidates:
        return None

    nearest_task = max(candidates, key=lambda position: position["y"])
    return nearest_task["task_num"]


def prepare_output_dir(output_dir):
    """Сохранить прежнюю выгрузку в input и создать пустой каталог."""
    if output_dir.exists():
        backup_root = INPUT_DIR / "previous_output"
        backup_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = backup_root / f"{output_dir.name}_{timestamp}"
        output_dir.rename(backup_dir)
        print(f"Предыдущая выгрузка сохранена: {backup_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def extract_task_images(pdf_path, tasks, images_dir):
    """Сохранить только встроенные изображения, связанные с заданиями.

    Привязка рассчитана на текущую вёрстку PDF с двумя колонками.
    QR-код и пример заполнения ответа находятся вне областей заданий
    и поэтому не сохраняются. Снимки целых заданий не создаются.
    """
    task_images = {task["task_num"]: [] for task in tasks}
    image_files = {}

    with pymupdf.open(pdf_path) as document:
        for page in document:
            positions = get_task_positions(page)
            middle_x = page.rect.width / 2

            for image in page.get_images():
                xref = image[0]
                related_tasks = set()

                for rect in page.get_image_rects(xref):
                    center_x = (rect.x0 + rect.x1) / 2
                    center_y = (rect.y0 + rect.y1) / 2
                    task_num = find_image_task(
                        positions, center_x, center_y, middle_x
                    )

                    if task_num in task_images:
                        related_tasks.add(task_num)

                # У служебных картинок нет подходящего задания.
                if not related_tasks:
                    continue

                if xref not in image_files:
                    image_data = document.extract_image(xref)
                    image_name = f"image_{xref}.{image_data['ext']}"
                    (images_dir / image_name).write_bytes(image_data["image"])
                    image_files[xref] = image_name

                image_name = image_files[xref]
                for task_num in sorted(related_tasks):
                    if image_name not in task_images[task_num]:
                        task_images[task_num].append(image_name)

    for task in tasks:
        names = task_images[task["task_num"]]
        task["image_name"] = "\n".join(names)
        if names:
            print(f"Задание {task['task_num']}: {task['image_name']}")

    print(f"Сохранено рисунков заданий: {len(image_files)}")


def validate_tasks(tasks, images_dir):
    """Проверить структуру данных и наличие указанных файлов.

    Пустое image_name допустимо: не у каждого задания есть рисунок.
    Эта проверка не оценивает правильность извлечения формул.
    """
    numbers = [task["task_num"] for task in tasks]
    if numbers != list(range(1, 20)):
        raise ValueError("Ожидались задания с номерами от 1 до 19")

    used_images = set()

    for task in tasks:
        number = task["task_num"]
        condition = task.get("condition")
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(f"У задания {number} пустое условие")

        answer = task.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"У задания {number} отсутствует текстовый ответ")

        for image_name in task["image_name"].splitlines():
            if not (images_dir / image_name).is_file():
                raise FileNotFoundError(
                    f"Задание {number}: не найден файл {image_name}"
                )
            used_images.add(image_name)

    saved_images = {
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.name != "tasks.xlsx"
    }
    if saved_images != used_images:
        raise ValueError(
            "Состав папки результата не совпадает "
            "с именами рисунков в заданиях"
        )

    print("Структура проверена: 19 заданий, ответы и ссылки на рисунки.")


def main(page_url):
    """Загрузить исходники и сформировать Excel, рисунки и ZIP."""
    INPUT_DIR.mkdir(exist_ok=True)
    variant_id = Path(urlsplit(page_url).path).stem
    if not variant_id:
        raise ValueError("В ссылке отсутствует имя страницы варианта")
    output_dir = BASE_DIR / "output" / variant_id

    html_data = download_bytes(page_url)
    html = html_data.decode("windows-1251")
    html_path = INPUT_DIR / f"{variant_id}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML сохранён: {html_path}")

    pdf_match = re.search(
        r'href\s*=\s*["\']([^"\']+\.pdf)["\']',
        html,
        flags=re.IGNORECASE,
    )
    if pdf_match is None:
        raise ValueError("На странице не найдена ссылка на PDF")

    pdf_url = urljoin(page_url, pdf_match.group(1))
    pdf_path = INPUT_DIR / Path(unquote(urlsplit(pdf_url).path)).name

    if pdf_path.is_file():
        print(f"Используем сохранённый PDF: {pdf_path}")
    else:
        pdf_data = download_bytes(pdf_url)

        if not pdf_data.startswith(b"%PDF-"):
            raise ValueError("Скачанный файл не похож на PDF")

        pdf_path.write_bytes(pdf_data)
        print(f"PDF сохранён: {pdf_path}")

    with pymupdf.open(pdf_path) as document:
        print(f"Количество страниц: {len(document)}")
        full_text = "\n".join(page.get_text() for page in document)

    if not full_text.strip():
        raise ValueError("Не удалось извлечь текст из PDF")

    text_path = INPUT_DIR / f"{pdf_path.stem}_raw_text.txt"
    text_path.write_text(full_text, encoding="utf-8")
    conditions = recognize_conditions(pdf_path)

    # Регулярка проверяет нумерацию. Текст условий берём из OCR.
    tasks = parse_tasks(full_text)
    for task in tasks:
        number = task["task_num"]
        try:
            task["condition"] = normalize_condition(conditions[number])
        except ValueError as error:
            raise ValueError(f"Задание {number}: {error}") from error

    print("\nЗадание 2 после распознавания:")
    print(tasks[1]["condition"])
    print(f"Выделено заданий: {len(tasks)}")

    script_sources = re.findall(
        r'<script\b[^>]*\bsrc\s*=\s*["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )
    expected_js_name = f"{pdf_path.stem}.js"
    js_url = None
    for source in script_sources:
        decoded_source = unquote(source)
        if Path(urlsplit(decoded_source).path).name == expected_js_name:
            js_url = urljoin(page_url, decoded_source)
            break

    if js_url is None:
        raise ValueError("Не найден JavaScript-файл этого варианта")

    js_data = download_bytes(js_url)
    js_path = INPUT_DIR / expected_js_name
    js_path.write_bytes(js_data)
    js_text = js_data.decode("utf-16")
    unpacked_js = packer.unpack(js_text) if packer.detect(js_text) else js_text
    readable_js = jsbeautifier.beautify(unpacked_js)
    unpacked_path = INPUT_DIR / f"{pdf_path.stem}_unpacked.js"
    unpacked_path.write_text(readable_js, encoding="utf-8")
    answers = parse_js_answers(readable_js)
    print(f"Из JavaScript извлечено ответов: {len(answers)}")

    image_sources = re.findall(
        r'<img\b[^>]*\bsrc\s*=\s*["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )
    answer_image_name = f"{pdf_path.stem}.png"
    answer_image_url = None
    for source in image_sources:
        decoded_source = unquote(source)
        if Path(urlsplit(decoded_source).path).name == answer_image_name:
            answer_image_url = urljoin(page_url, decoded_source)
            break

    if answer_image_url is None:
        raise ValueError("Не найдена картинка с ответами этого варианта")

    answer_image_path = INPUT_DIR / answer_image_name
    answer_image_path.write_bytes(download_bytes(answer_image_url))

    extra_answers_path = INPUT_DIR / f"{pdf_path.stem}_answers.json"
    extra_data = json.loads(extra_answers_path.read_text(encoding="utf-8"))
    if extra_data["source_url"] != answer_image_url:
        raise ValueError("JSON относится к другой картинке ответов")

    extra_answers = {}
    for number, value in extra_data["answers"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Пустой или некорректный ответ: {number}")
        task_num = int(number)
        if task_num in extra_answers:
            raise ValueError(f"Повторный номер ответа: {task_num}")
        extra_answers[task_num] = value

    if sorted(extra_answers) != list(range(13, 20)):
        raise ValueError("В дополнительном файле нужны ответы 13–19")

    answers.update(extra_answers)
    for task in tasks:
        task["answer"] = answers.get(task["task_num"], "")
    print(f"Всего ответов: {len(answers)}")

    metadata = parse_metadata(full_text, page_url, date.today())
    metadata["answers_13_19_method"] = extra_data["method"]
    metadata["answers_13_19_source"] = extra_data["source_url"]
    metadata["conditions_method"] = METHOD
    metadata["conditions_model"] = MODEL
    metadata["conditions_dpi"] = DPI
    metadata["conditions_format"] = "LaTeX для отображения через MathJax"
    metadata["conditions_cache_format"] = "LaTeX"
    metadata["formula_notation"] = (
        "$...$ — формула внутри текста; "
        "$$...$$ — отдельная формула"
    )
    metadata["conditions_source"] = pdf_url

    classifications = classify_tasks(
        tasks,
        BASE_DIR / "data" / "exams.csv",
        BASE_DIR / "data" / "topics.csv",
        INPUT_DIR / f"{pdf_path.stem}_classification.json",
    )
    for task in tasks:
        task.update(classifications[task["task_num"]])
        task["condition"] = add_subtask_breaks(task["condition"])
        task["answer"] = add_subtask_breaks(task["answer"])
    print(f"Классификация получена для {len(classifications)} заданий.")

    images_dir = prepare_output_dir(output_dir)
    extract_task_images(pdf_path, tasks, images_dir)
    validate_tasks(tasks, images_dir)

    excel_path = output_dir / "tasks.xlsx"
    write_excel(tasks, prepare_about_metadata(metadata), excel_path)
    print(f"Excel сохранён: {excel_path}")

    archive_path = BASE_DIR / f"{pdf_path.stem}_result.zip"
    file_count = create_zip(excel_path, images_dir, archive_path)
    print(f"Архив сохранён вне output: {archive_path}")
    print(f"Файлов в архиве: {file_count}")
    print("Условия распознаны по изображениям через Gemini. Требуется сверка с PDF.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Сбор тренировочного варианта alexlarin.net"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_PAGE_URL,
        help="Ссылка на страницу варианта",
    )
    args = parser.parse_args()
    main(args.url)
