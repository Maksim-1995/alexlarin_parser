"""Собирает задания и ответы alexlarin.net в Excel и ZIP-архив.

Исходные файлы сохраняются в input, результаты — в output/<variant_id>.
Ответы на задания 13–19 читаются из подготовленного JSON-файла.
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import jsbeautifier
import pymupdf
from jsbeautifier.unpackers import packer

from downloader import download_bytes
from exporter import create_zip, write_excel
from parsers import TASK_RE, parse_js_answers, parse_tasks

DEFAULT_PAGE_URL = "https://alexlarin.net/ege/2026/trvar542.html"

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"


def parse_metadata(text, source_url, export_date):
    """Извлечь класс, год экзамена и название варианта из текста PDF.

    Args:
        text (str): Полный текст тренировочного варианта.
        source_url (str): Ссылка на страницу варианта.
        export_date (datetime.date): Дата формирования выгрузки.

    Returns:
        dict: Класс, год, название варианта, URL и дата выгрузки
            с пояснением её назначения в поле date_kind.

    Raises:
        ValueError: Не найден класс, год экзамена или номер варианта.
    """
    class_match = re.search(
        r"\b(\d{1,2})\s*класс\b",
        text,
        flags=re.IGNORECASE,
    )

    year_match = re.search(
        r"экзамен,\s*(\d{4})\s*г\b",
        text,
        flags=re.IGNORECASE,
    )

    variant_match = re.search(
        r"Тренировочный\s+вариант\s*№\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if not (class_match and year_match and variant_match):
        raise ValueError(
            "Не удалось найти класс, год или номер варианта"
        )

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


def get_task_positions(page):
    """Найти номера заданий в начале строк на странице PDF.

    Args:
        page (pymupdf.Page): Страница с текстовым слоем.

    Returns:
        list[dict]: Номера заданий в поле task_num и координаты
            левого верхнего угла номера в полях x и y.
    """
    positions = []

    for word_data in page.get_text("words"):
        x0, y0, x1, y1, word, block_num, line_num, word_num = word_data

        if word_num != 0:
            continue

        match = TASK_RE.fullmatch(word)

        if match is None:
            continue

        positions.append({
            "task_num": int(match.group(1)),
            "x": x0,
            "y": y0,
        })

    return positions


def find_image_task(positions, center_x, center_y, middle_x):
    """Определить задание для изображения по колонке и высоте.

    Args:
        positions (list[dict]): Номера заданий и их координаты x, y.
        center_x (float): Горизонтальная координата центра изображения.
        center_y (float): Вертикальная координата центра изображения.
        middle_x (float): Горизонтальная граница двух колонок страницы.

    Returns:
        int | None: Номер ближайшего задания в той же колонке,
            начинающегося не ниже центра изображения, или None.
    """
    image_is_right = center_x >= middle_x
    candidates = []

    for position in positions:
        task_is_right = position["x"] >= middle_x

        same_column = task_is_right == image_is_right
        starts_above = position["y"] <= center_y

        if same_column and starts_above:
            candidates.append(position)

    if not candidates:
        return None

    nearest_task = max(
        candidates,
        key=lambda position: position["y"],
    )

    return nearest_task["task_num"]


def get_task_areas(page, top_padding=12, footer_margin=25):
    """Вычислить области снимков заданий в двух колонках страницы PDF.

    Область заканчивается перед следующим заданием той же колонки
    или перед нижним колонтитулом. Отступы задаются в пунктах PDF.

    Args:
        page (pymupdf.Page): Страница с текстовым слоем.
        top_padding (float): Отступ над номером задания.
        footer_margin (float): Отступ от нижнего края страницы.

    Returns:
        dict[int, pymupdf.Rect]: Области снимков по номерам заданий.
    """
    positions = get_task_positions(page)
    middle_x = page.rect.width / 2
    areas = {}

    column_bounds = [
        (0, middle_x),
        (middle_x, page.rect.width),
    ]

    for left, right in column_bounds:
        column_tasks = [
            position
            for position in positions
            if left <= position["x"] < right
        ]

        column_tasks.sort(key=lambda position: position["y"])

        for index, position in enumerate(column_tasks):
            top = max(0, position["y"] - top_padding)

            if index + 1 < len(column_tasks):
                bottom = column_tasks[index + 1]["y"] - top_padding
            else:
                bottom = page.rect.height - footer_margin

            areas[position["task_num"]] = pymupdf.Rect(
                left,
                top,
                right,
                bottom,
            )

    return areas


def validate_tasks(tasks, images_dir):
    """Проверить задания 1–19, текстовые ответы и наличие изображений.

    Args:
        tasks (list[dict]): Задания по порядку с полями task_num,
            condition, answer и image_name. Имена файлов разделены
            переводами строк.
        images_dir (pathlib.Path): Каталог изображений заданий.

    Raises:
        ValueError: Нарушен порядок номеров или отсутствуют условие,
            текстовый ответ либо имя изображения.
        FileNotFoundError: Указанный файл изображения не существует.
    """
    numbers = [task["task_num"] for task in tasks]

    if numbers != list(range(1, 20)):
        raise ValueError("Ожидались задания с номерами от 1 до 19")

    for task in tasks:
        number = task["task_num"]

        if not task["condition"].strip():
            raise ValueError(f"У задания {number} пустое условие")

        answer = task.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"У задания {number} отсутствует текстовый ответ")

        image_names = task["image_name"].splitlines()

        if not image_names:
            raise ValueError(f"У задания {number} отсутствует снимок")

        for image_name in image_names:
            image_path = images_dir / image_name

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Задание {number}: не найден файл {image_path}"
                )

    print("Проверка пройдена: 19 заданий, ответы и изображения на месте.")


INPUT_DIR.mkdir(exist_ok=True)


def main(page_url):
    """Загрузить вариант и создать Excel, изображения и ZIP-архив.

    Скачивает HTML, PDF, JavaScript и картинку с ответами. Ответы 13–19
    читает из input/<имя PDF>_answers.json. Сохраняет результаты
    в output/<идентификатор страницы> и проверяет полноту.

    Args:
        page_url (str): URL страницы тренировочного варианта.

    Raises:
        requests.RequestException: Не удалось загрузить исходные файлы.
        FileNotFoundError: Нет дополнительного JSON или изображения.
        ValueError: Исходные данные или результаты не прошли проверку.
    """
    INPUT_DIR.mkdir(exist_ok=True)

    variant_id = Path(urlsplit(page_url).path).stem

    output_dir = BASE_DIR / "output" / variant_id
    images_dir = output_dir / "images"

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

    pdf_link = pdf_match.group(1)
    pdf_url = urljoin(page_url, pdf_link)

    print(f"Найдена ссылка на PDF: {pdf_url}")

    pdf_data = download_bytes(pdf_url)

    if not pdf_data.startswith(b"%PDF-"):
        raise ValueError("Скачанный файл не похож на PDF")

    pdf_name = Path(pdf_link).name
    pdf_path = INPUT_DIR / pdf_name
    pdf_path.write_bytes(pdf_data)

    print(f"PDF сохранён: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    page_texts = []

    with pymupdf.open(pdf_path) as document:
        print(f"Количество страниц: {len(document)}")

        for page in document:
            page_text = page.get_text()
            page_texts.append(page_text)

    full_text = "\n".join(page_texts)

    if not full_text.strip():
        raise ValueError("Не удалось извлечь текст из PDF")

    text_path = output_dir / "raw_text.txt"
    text_path.write_text(full_text, encoding="utf-8")

    print(f"Извлечено символов: {len(full_text)}")
    print(f"Текст сохранён: {text_path}")

    tasks = parse_tasks(full_text)

    task_numbers = [task["task_num"] for task in tasks]
    print(f"Найдены номера: {task_numbers}")
    print(f"Выделено заданий: {len(tasks)}")

    for task_index in (0, 11, 18):
        task = tasks[task_index]

        print(f"\nЗадание {task['task_num']}:")
        print(task["condition"])

    images_dir.mkdir(exist_ok=True)

    saved_xrefs = set()
    image_files = {}

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
                image_path = images_dir / image_name

                image_path.write_bytes(image_data["image"])
                image_files[xref] = image_name
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
            js_url = urljoin(page_url, decoded_source)
            break

    if js_url is None:
        raise ValueError("Не найден JavaScript-файл этого варианта")

    print(f"Найдена ссылка на JavaScript: {js_url}")

    js_data = download_bytes(js_url)
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

    unpacked_path = output_dir / f"{pdf_path.stem}_unpacked.js"
    unpacked_path.write_text(readable_js, encoding="utf-8")

    print(f"Распакованный JavaScript сохранён: {unpacked_path}")

    answers = parse_js_answers(readable_js)

    print(f"Извлечено ответов: {len(answers)}")

    for number, value in sorted(answers.items()):
        print(f"{number}: {value}")

    image_sources = re.findall(
        r'<img\b[^>]*\bsrc\s*=\s*["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )

    answer_image_name = f"{pdf_path.stem}.png"
    answer_image_url = None

    for source in image_sources:
        decoded_source = unquote(source)

        if Path(decoded_source).name == answer_image_name:
            answer_image_url = urljoin(page_url, decoded_source)
            break

    if answer_image_url is None:
        raise ValueError("Не найдена картинка с ответами этого варианта")

    print(f"Найдена картинка с ответами: {answer_image_url}")

    answer_image_data = download_bytes(answer_image_url)

    answer_image_path = INPUT_DIR / answer_image_name
    answer_image_path.write_bytes(answer_image_data)

    print(f"Картинка с ответами сохранена: {answer_image_path}")

    extra_answers_path = INPUT_DIR / f"{pdf_path.stem}_answers.json"

    extra_data = json.loads(
        extra_answers_path.read_text(encoding="utf-8")
    )

    if extra_data["source_url"] != answer_image_url:
        raise ValueError("JSON относится к другой картинке ответов")

    extra_answers = {}

    for number, value in extra_data["answers"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Пустой или некорректный ответ: {number}")

        extra_answers[int(number)] = value

    if sorted(extra_answers) != list(range(13, 20)):
        raise ValueError("В дополнительном файле нужны ответы 13–19")

    answers.update(extra_answers)

    for task in tasks:
        task["answer"] = answers.get(task["task_num"], "")

    print(f"Всего ответов: {len(answers)}")

    for number in range(13, 20):
        print(f"{number}: {answers[number]}")

    task_images = {
        task["task_num"]: []
        for task in tasks
    }

    with pymupdf.open(pdf_path) as document:
        for page in document:
            middle_x = page.rect.width / 2
            positions = get_task_positions(page)

            for image in page.get_images():
                xref = image[0]

                for rect in page.get_image_rects(xref):
                    center_x = (rect.x0 + rect.x1) / 2
                    center_y = (rect.y0 + rect.y1) / 2

                    task_num = find_image_task(
                        positions,
                        center_x,
                        center_y,
                        middle_x,
                    )

                    if task_num is None:
                        continue

                    image_name = image_files[xref]

                    if image_name not in task_images[task_num]:
                        task_images[task_num].append(image_name)

    for task in tasks:
        task["image_name"] = "\n".join(
            task_images[task["task_num"]]
        )

        if task["image_name"]:
            print(
                f"Задание {task['task_num']}: "
                f"{task['image_name']}"
            )

    saved_previews = set()

    with pymupdf.open(pdf_path) as document:
        for page in document:
            areas = get_task_areas(page)

            for task_num, area in areas.items():
                preview_name = f"task_{task_num:02d}.png"
                preview_path = images_dir / preview_name

                pixmap = page.get_pixmap(
                    clip=area,
                    dpi=150,
                )
                pixmap.save(preview_path)

                if preview_name not in task_images[task_num]:
                    task_images[task_num].insert(0, preview_name)

                saved_previews.add(task_num)

    if saved_previews != set(range(1, 20)):
        raise ValueError("Не удалось сохранить снимки всех заданий 1–19")

    for task in tasks:
        task["image_name"] = "\n".join(
            task_images[task["task_num"]]
        )

    print(f"Сохранено снимков заданий: {len(saved_previews)}")
    print("Файлы задания 1:", tasks[0]["image_name"].splitlines())

    metadata = parse_metadata(
        full_text,
        page_url,
        date.today(),
    )

    metadata["answers_13_19_method"] = extra_data["method"]
    metadata["answers_13_19_source"] = extra_data["source_url"]

    print("\nМетаданные варианта:")

    for field, value in metadata.items():
        print(f"{field}: {value}")

    excel_path = output_dir / "tasks.xlsx"

    validate_tasks(tasks, images_dir)
    write_excel(tasks, metadata, excel_path)

    print(f"Excel сохранён: {excel_path}")

    archive_path = output_dir / f"{pdf_path.stem}_result.zip"

    file_count = create_zip(
        excel_path,
        images_dir,
        archive_path,
    )

    print(f"Архив сохранён: {archive_path}")
    print(f"Файлов в архиве: {file_count}")


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
