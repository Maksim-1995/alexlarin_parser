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
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import jsbeautifier
import pymupdf
from jsbeautifier.unpackers import UnpackingError, packer

from classification import classify_tasks
from downloader import load_source
from exporter import create_zip, write_excel
from export_metadata import prepare_about_metadata
from latex_format import normalize_condition
from parsers import parse_js_answers, parse_tasks
from recognizer import DPI, METHOD, MODEL, get_task_positions, recognize_conditions
from task_html import add_subtask_breaks


DEFAULT_PAGE_URL = "https://alexlarin.net/ege/2026/trvar542.html"
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"


def parse_source_page(data, page_url):
    """Прочитать сохранённый UTF-8 HTML или исходный windows-1251 HTML.

    Ссылки разрешаются относительно адреса alexlarin.net, а не input/.
    Проверка номера и ссылок отсеивает страницы ошибок и другие варианты.
    """
    try:
        html = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        html = data.decode("windows-1251")
    variant_id = Path(urlsplit(page_url).path).stem
    variant_match = re.fullmatch(r"trvar(\d+)", variant_id)
    title_match = re.search(r"<title\b[^>]*>(.*?)</title\s*>", html, re.I | re.S)
    title = unescape(title_match.group(1)) if title_match else ""
    number_match = re.search(r"Тренировочный\s+вариант\s*№\s*(\d+)", title, re.I)
    if not variant_match or not number_match or number_match.group(1) != variant_match.group(1):
        raise ValueError("HTML не содержит заголовок нужного тренировочного варианта")
    expected_year = Path(urlsplit(page_url).path).parent.name
    if expected_year.isdigit() and not re.search(rf"\b{expected_year}\b", title):
        raise ValueError("Год в HTML не соответствует исходной ссылке")

    patterns = {
        "pdf": r'href\s*=\s*["\']([^"\']+)["\']',
        "js": r'<script\b[^>]*\bsrc\s*=\s*["\']([^"\']+)["\']',
        "png": r'<img\b[^>]*\bsrc\s*=\s*["\']([^"\']+)["\']',
    }
    links = {}
    source_address = urlsplit(page_url)
    for extension, pattern in patterns.items():
        expected_name = f"{variant_id}.{extension}"
        expected_path = urlsplit(urljoin(page_url, expected_name)).path
        for source in re.findall(pattern, html, re.I):
            decoded_source = unquote(unescape(source))
            resource_url = urljoin(page_url, decoded_source)
            address = urlsplit(resource_url)
            if (
                address.scheme in {"http", "https"}
                and address.netloc.casefold() == source_address.netloc.casefold()
                and address.path == expected_path
            ):
                links[extension] = resource_url
                break
        if extension not in links:
            raise ValueError(f"В HTML не найдена ссылка на {expected_name}")
    return links


def read_pdf_source(data, page_url):
    """Проверить PDF, номер варианта, год и полноту текстового слоя."""
    if not data.startswith(b"%PDF-"):
        raise ValueError("Файл не похож на PDF")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            full_text = "\n".join(page.get_text() for page in document)
    except (RuntimeError, ValueError) as error:
        raise ValueError("Не удалось прочитать PDF") from error
    metadata = parse_metadata(full_text, page_url, date.today())
    expected_number = re.fullmatch(r"trvar(\d+)", Path(urlsplit(page_url).path).stem)
    if not expected_number or metadata["variant_name"] != f"Тренировочный вариант № {expected_number.group(1)}":
        raise ValueError("PDF относится к другому варианту")
    expected_year = Path(urlsplit(page_url).path).parent.name
    if expected_year.isdigit() and metadata["year"] != int(expected_year):
        raise ValueError("Год в PDF не соответствует исходной ссылке")
    parse_tasks(full_text)
    return full_text


def read_js_source(data):
    """Распаковать JavaScript и проверить наличие всех ответов 1–12."""
    try:
        js_text = data.decode("utf-16") if data.startswith((b"\xff\xfe", b"\xfe\xff")) else data.decode("utf-8-sig")
        unpacked_js = packer.unpack(js_text) if packer.detect(js_text) else js_text
        readable_js = jsbeautifier.beautify(unpacked_js)
        answers = parse_js_answers(readable_js)
    except (UnicodeError, RuntimeError, ValueError, IndexError, UnpackingError) as error:
        raise ValueError("JavaScript не содержит корректные ответы 1–12") from error
    return readable_js, answers


def validate_answer_image(data):
    """Проверить PNG с опубликованными ответами через декодирование."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Файл не похож на PNG с ответами")
    try:
        image = pymupdf.Pixmap(data)
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Пустое изображение")
    except (RuntimeError, ValueError) as error:
        raise ValueError("Не удалось прочитать PNG с ответами") from error


def read_extra_answers(path, image_url):
    """Прочитать ручной перенос ответов 13–19 и проверить его источник."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"Не удалось прочитать файл ответов {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"В файле ответов {path} нужен JSON-объект")
    if data.get("source_url") != image_url:
        raise ValueError(f"JSON ответов {path} относится к другой картинке ответов")
    if not isinstance(data.get("method"), str) or not data["method"].strip():
        raise ValueError(f"В файле ответов {path} не указан способ переноса method")
    if not isinstance(data.get("answers"), dict):
        raise ValueError(f"В файле ответов {path} нужен объект answers с ответами 13–19")

    answers = {}
    for number, value in data["answers"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Пустой или некорректный ответ {number} в {path}")
        try:
            task_num = int(number)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Некорректный номер ответа {number!r} в {path}") from error
        if task_num in answers:
            raise ValueError(f"Повторный номер ответа {task_num} в {path}")
        answers[task_num] = value
    if sorted(answers) != list(range(13, 20)):
        raise ValueError(f"В файле {path} нужны ровно ответы 13–19")
    return {**data, "answers": answers}


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

    html_path = INPUT_DIR / f"{variant_id}.html"
    source_links = load_source(
        page_url, html_path, lambda data: parse_source_page(data, page_url)
    )
    pdf_url = source_links["pdf"]
    pdf_path = INPUT_DIR / Path(unquote(urlsplit(pdf_url).path)).name
    full_text = load_source(
        pdf_url, pdf_path, lambda data: read_pdf_source(data, page_url)
    )

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

    expected_js_name = f"{pdf_path.stem}.js"
    js_path = INPUT_DIR / expected_js_name
    readable_js, answers = load_source(source_links["js"], js_path, read_js_source)
    unpacked_path = INPUT_DIR / f"{pdf_path.stem}_unpacked.js"
    unpacked_path.write_text(readable_js, encoding="utf-8")
    print(f"Из JavaScript извлечено ответов: {len(answers)}")
    answer_image_name = f"{pdf_path.stem}.png"
    answer_image_url = source_links["png"]
    answer_image_path = INPUT_DIR / answer_image_name
    load_source(answer_image_url, answer_image_path, validate_answer_image)

    extra_answers_path = INPUT_DIR / f"{pdf_path.stem}_answers.json"
    extra_data = read_extra_answers(extra_answers_path, answer_image_url)
    answers.update(extra_data["answers"])
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
