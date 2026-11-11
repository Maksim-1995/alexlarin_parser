"""Прежние проверки источников современного варианта 542.

Эти чистые функции сохраняют принятые строгие правила старого запуска:
19 задач, JS-ответы 1–12 и ручной перенос 13–19. Они вынесены отдельно,
чтобы ограничения 542 не становились правилами общего пакетного потока.
Новый поток работает через variant_sources.py; main.py повторно экспортирует
прежние имена для совместимости скриптов и локальных регрессионных проверок.
"""

import json
import re
from datetime import date
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import jsbeautifier
import pymupdf
from jsbeautifier.unpackers import UnpackingError, packer

from parsers import parse_js_answers, parse_tasks


def parse_source_page(data, page_url):
    """Прочитать сохранённый UTF-8 HTML или исходный windows-1251 HTML.

    Ссылки разрешаются относительно адреса alexlarin.net, а не input/.
    Проверка номера и ссылок отсеивает страницы ошибок и другие варианты.

    Args:
        data: Байты исходного HTML в UTF-8 или windows-1251.
        page_url: Исходный URL для проверки номера, года и адресов файлов.

    Returns:
        Словарь с абсолютными URL исходников в ключах pdf, js и png.

    Raises:
        ValueError: Заголовок или ссылки не соответствуют нужному варианту.
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
    """Проверить PDF, номер варианта, год и полноту текстового слоя.

    Args:
        data: Байты PDF, а не путь или результат текстового извлечения.
        page_url: Исходный URL страницы с ожидаемым номером и годом.

    Returns:
        Полный текстовый слой всех страниц после прежних проверок 542.

    Raises:
        ValueError: PDF повреждён, относится к другому варианту или не
            содержит полного принятого набора задач 1–19.
    """
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
    """Распаковать JavaScript и проверить наличие всех ответов 1–12.

    Args:
        data: Байты опубликованного JS в UTF-8 либо UTF-16 с BOM.

    Returns:
        Пара из читаемого JS и словаря числовых номеров с текстами ответов.

    Raises:
        ValueError: Кодировка, упаковка или набор ответов некорректны.
    """
    try:
        js_text = data.decode("utf-16") if data.startswith((b"\xff\xfe", b"\xfe\xff")) else data.decode("utf-8-sig")
        unpacked_js = packer.unpack(js_text) if packer.detect(js_text) else js_text
        readable_js = jsbeautifier.beautify(unpacked_js)
        answers = parse_js_answers(readable_js)
    except (UnicodeError, RuntimeError, ValueError, IndexError, UnpackingError) as error:
        raise ValueError("JavaScript не содержит корректные ответы 1–12") from error
    return readable_js, answers


def validate_answer_image(data):
    """Проверить PNG с опубликованными ответами через декодирование.

    Args:
        data: Байты PNG, включая сигнатуру файла.

    Returns:
        None после успешного декодирования непустого изображения.

    Raises:
        ValueError: Содержимое не является корректным непустым PNG.
    """
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Файл не похож на PNG с ответами")
    try:
        image = pymupdf.Pixmap(data)
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Пустое изображение")
    except (RuntimeError, ValueError) as error:
        raise ValueError("Не удалось прочитать PNG с ответами") from error


def read_extra_answers(path, image_url):
    """Прочитать ручной перенос ответов 13–19 и проверить его источник.

    Args:
        path: Путь к JSON с source_url, method и словарём answers.
        image_url: Точный URL опубликованной картинки, с которой перенесены
            ответы. Ответы другого варианта не допускаются.

    Returns:
        Данные исходного JSON с целочисленными ключами answers 13–19.
        Сами ответы сохраняются строками без математических преобразований.

    Raises:
        ValueError: Файл недоступен, принадлежит другому источнику или
            содержит неполные, повторные либо некорректные ответы.
    """
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
    """Извлечь класс, год экзамена и название варианта из текста PDF.

    Args:
        text: Полный текстовый слой PDF современного формата.
        source_url: Исходный URL страницы, сохраняемый в метаданных.
        export_date: Явно переданная дата формирования старой выгрузки;
            эта функция не трактует её как дату публикации варианта.

    Returns:
        Прежний словарь class, year, date, variant_name, source_url и
        date_kind. Состав сохранён для совместимости старых проверок.

    Raises:
        ValueError: В тексте не найдены класс, год или номер варианта.
    """
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
