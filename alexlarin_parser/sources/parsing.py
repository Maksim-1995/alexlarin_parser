"""Разбор и проверка содержимого HTML, PDF, JS и изображений без загрузки.

Здесь нормализуются исходные метки и видимые координаты. Модуль проверяет
содержимое конкретного варианта, но не управляет манифестом или кэшами.
"""

from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlsplit

import jsbeautifier
from jsbeautifier.unpackers import packer, UnpackingError
import pymupdf

from ..context import _canonical_url


LABEL_RE = re.compile(r"^([BВCС]?)[ \t]*(\d+)[.)]?$", re.I)


def normalize_label(label):
    """B/В и C/С — явные визуальные эквиваленты, остальные буквы запрещены."""
    if not isinstance(label, str):
        raise ValueError(f"Метка задачи должна быть строкой: {label!r}")
    match = LABEL_RE.fullmatch(label.strip())
    if not match or int(match[2]) <= 0:
        raise ValueError(f"Некорректная метка основной задачи: {label!r}")
    letter = match[1].upper().translate(str.maketrans({"В": "B", "С": "C"}))
    return f"{letter}{int(match[2])}"


def decode_html(data):
    """Декодировать сохранённый UTF-8 HTML или старую страницу Windows-1251."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("windows-1251")


class SourceHTMLParser(HTMLParser):
    """Ссылки и строки таблиц, включая вложенные таблицы старых страниц."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.rows = []
        self._tables = []
        self._title_depth = 0
        self.title_parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self._title_depth += 1
        if tag in {"a", "object", "script", "img"}:
            source = attrs.get({"a": "href", "object": "data", "script": "src", "img": "src"}[tag])
            if source:
                self.links.append((tag, source))
        if tag == "table":
            self._tables.append({"row": None, "cell": None})
        elif tag == "tr" and self._tables:
            self._tables[-1].update(row=[], cell=None)
        elif tag in {"td", "th"} and self._tables:
            self._tables[-1]["cell"] = []
        elif tag == "br" and self._tables and self._tables[-1]["cell"] is not None:
            self._tables[-1]["cell"].append(" ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._title_depth = max(0, self._title_depth - 1)
        if tag in {"td", "th"} and self._tables:
            table = self._tables[-1]
            if table["row"] is not None and table["cell"] is not None:
                table["row"].append("".join(table["cell"]))
            table["cell"] = None
        elif tag == "tr" and self._tables:
            table = self._tables[-1]
            if table["row"]:
                self.rows.append(table["row"])
            table.update(row=None, cell=None)
        elif tag == "table" and self._tables:
            self._tables.pop()

    def handle_data(self, data):
        if self._title_depth:
            self.title_parts.append(data)
        if self._tables and self._tables[-1]["cell"] is not None:
            self._tables[-1]["cell"].append(data)

    @property
    def title(self):
        """Вернуть заголовок страницы с нормализованными пробелами."""
        return " ".join("".join(self.title_parts).split())


def parse_variant_html(data, context):
    """Проверить номер/год HTML и извлечь ссылки и ответы таблиц B/C.

    Принимаются только ресурсы того же кода в каталоге текущей страницы.
    Ответы сохраняются строками; исходные буквенные метки возвращаются
    отдельно от канонических ключей B/C для последующей проверки.
    """
    if not data.strip() or b"<" not in data:
        raise ValueError("Пустой исходник HTML")
    parser = SourceHTMLParser()
    parser.feed(decode_html(data))
    number = re.search(r"Тренировочный\s+вариант\s*№\s*(\d+)", parser.title, re.I)
    if not number or int(number[1]) != context.number:
        raise ValueError("HTML относится к другому варианту или является страницей ошибки")
    if not re.search(rf"\b{context.year}\b", parser.title):
        raise ValueError("Год HTML не соответствует исходной ссылке")
    resources = {"pdf": [], "js": [], "images": []}
    page_parent = Path(urlsplit(context.source_url).path).parent
    for tag, href in parser.links:
        try:
            url = _canonical_url(urljoin(context.source_url, unescape(href)))
        except ValueError:
            continue
        path = Path(unquote(urlsplit(url).path))
        # Объявления, эмблемы и картинки соседнего варианта не являются ответами.
        if path.parent != page_parent or path.stem != context.code:
            continue
        if path.suffix.lower() == ".pdf" and tag in {"a", "object"}:
            resources["pdf"].append(url)
        elif path.suffix.lower() == ".js" and tag == "script":
            resources["js"].append(url)
        elif path.suffix.lower() in {".png", ".jpg", ".jpeg"} and tag in {"img", "a"}:
            resources["images"].append(url)
    for key in resources:
        resources[key] = list(dict.fromkeys(resources[key]))
    if len(resources["pdf"]) != 1:
        raise ValueError("В HTML нет однозначной ссылки на PDF текущего варианта")
    answers = {}
    answer_labels = {}
    for row in parser.rows:
        if len(row) != 2:
            continue
        original = " ".join(row[0].split())
        try:
            label = normalize_label(original)
        except ValueError:
            continue
        # Числа в рекламных и версточных таблицах не считаются ответами.
        if not label.startswith(("B", "C")):
            continue
        answer = " ".join(row[1].split())
        if not answer:
            continue
        if label in answers:
            raise ValueError(f"Неоднозначный HTML-ответ {label}")
        answers[label] = answer
        answer_labels[label] = original
    return {"title": parser.title, "resources": resources, "answers": answers, "answer_labels": answer_labels}


def task_positions(page):
    """Найти основные метки PDF и их координаты на отображаемой странице.

    Возвращает каноническую и исходную метки с x/y/x1/y1 после поворота PDF.
    Порядок соответствует меткам задач, а не порядку PDF text stream.
    Подпункты А), Б), В) без числа основными задачами не считаются.
    """
    positions = []
    words = page.get_text("words")
    visible_words = [pymupdf.Rect(*word[:4]) * page.rotation_matrix for word in words]
    label_columns = [visible_words[index].x0 for index, word in enumerate(words)
                     if word[7] == 0 and re.fullmatch(r"[BВCС]\d+[.)]?", word[4], re.I)]
    for index, word in enumerate(words):
        x0, y0, x1, y1, value, block, line, word_num = word
        if word_num != 0:
            continue
        candidate = value
        # В части старых PDF метка и первое русское слово слиты в один word
        # (например B13Первая), хотя визуально разделены расстоянием.
        fused = re.match(r"^([BВCС]\d+)(?=[А-ЯЁ])", value, re.I)
        if fused:
            candidate = fused[1]
        if value.upper() in {"B", "В", "C", "С"}:
            # Буква B и цифра 8 могут лежать в разных PDF text blocks.
            # Сопоставляем только непосредственно соседние видимые глифы
            # на одной базовой линии, а не произвольную цифру из text stream.
            anchor = visible_words[index]
            # Фраза «В 2008 году» тоже содержит соседние букву и цифры.
            # При известных целых метках раздельная метка должна находиться
            # на той же линии начал задач, а не внутри текста условия.
            if label_columns and not any(abs(anchor.x0 - x) <= 6 for x in label_columns):
                continue
            neighbors = [j for j, other in enumerate(words)
                         if re.fullmatch(r"\d+[.)]?", other[4])
                         and -1 <= visible_words[j].x0 - anchor.x1 <= 4
                         and abs((visible_words[j].y0 + visible_words[j].y1)
                                 - (anchor.y0 + anchor.y1)) <= 4]
            if len(neighbors) == 1:
                other = words[neighbors[0]]
                candidate += other[4]
                joined = pymupdf.Rect(x0, y0, x1, y1) | pymupdf.Rect(*other[:4])
                x0, y0, x1, y1 = joined
        # Современная цифровая метка должна иметь точку; иначе цифра формулы
        # или номер страницы ошибочно становится основной задачей.
        if not candidate[:1].upper() in {"B", "В", "C", "С"} and not candidate.endswith("."):
            continue
        try:
            label = normalize_label(candidate)
        except ValueError:
            continue
        # get_text/get_image_rects возвращают координаты до /Rotate, а
        # page.rect и OCR-рендер уже учитывают поворот старых PDF на 90°.
        rect = pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix
        positions.append({"source_label": label, "original_label": candidate,
                          "x": rect.x0, "y": rect.y0, "x1": rect.x1, "y1": rect.y1})
    if any(p["source_label"][0] in {"B", "C"} for p in positions):
        # В формате B/C цифра «1.» или «729.» внутри формулы не является
        # ещё одной основной задачей цифрового формата.
        positions = [p for p in positions if p["source_label"][0] in {"B", "C"}]
    # Порядок объектов content stream старых PDF не равен порядку чтения:
    # например C5 может быть записана перед B12. Канонические метки задают
    # стабильный порядок основных задач независимо от внутренней записи PDF.
    def label_order(position):
        label = position["source_label"]
        match = re.fullmatch(r"([BC]?)(\d+)", label)
        return ({"": 0, "B": 1, "C": 2}[match[1]], int(match[2]))
    return sorted(positions, key=label_order)


def inspect_pdf(data, context):
    """Проверить PDF и собрать метки, страницы и геометрию для OCR/рисунков.

    Количество задач определяется метками, а не вступительной инструкцией.
    ``page_labels`` содержит только первые появления задач; повторная метка
    продолжения остаётся в ``page_positions``. Недостаток текстового слоя
    возвращается предупреждением и требует последующей проверки по изображению.
    """
    if not data.startswith(b"%PDF-"):
        raise ValueError("Файл не похож на PDF")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if not len(document):
                raise ValueError("PDF не содержит страниц")
            texts = [page.get_text() for page in document]
            text = "\n".join(texts)
            number = re.search(r"(?:Тренировочный\s+)?вариант\s*№\s*(\d+)", text, re.I)
            if number and int(number[1]) != context.number:
                raise ValueError("PDF относится к другому варианту")
            # В старых PDF диапазон 2012‐2013 напечатан с U+2010, а не
            # ASCII-дефисом. Разделитель нормализуем только в поиске года:
            # иначе первая половина диапазона ошибочно считается годом ЕГЭ.
            year = re.search(r"(?:экзамен[,\s]+|ЕГЭ\s*[-–]?\s*)(20\d\d)(?:\s*[-‐‑‒–—−]\s*(20\d\d))?", text, re.I)
            exam_year = int(year[2] or year[1]) if year else None
            if year and exam_year != context.year:
                raise ValueError("PDF относится к другому году")
            page_positions = [task_positions(page) for page in document]
            page_dimensions = [{"width": page.rect.width, "height": page.rect.height} for page in document]
            printed_labels = [[position["source_label"] for position in positions] for positions in page_positions]
            seen_labels = set()
            page_labels = []
            for printed in printed_labels:
                starts = []
                for label in printed:
                    if label not in seen_labels:
                        starts.append(label)
                        seen_labels.add(label)
                page_labels.append(starts)
    except (RuntimeError, pymupdf.FileDataError) as error:
        raise ValueError("Не удалось прочитать PDF") from error
    labels = []
    for page in page_labels:
        for label in page:
            if label not in labels:
                labels.append(label)
    # Порядковая нумерация определяется фактическими метками, а не инструкцией.
    def order(label):
        match = re.fullmatch(r"([BC]?)(\d+)", label)
        return ({"": 0, "B": 1, "C": 2}[match[1]], int(match[2]))
    labels.sort(key=order)
    warnings = []
    if not number:
        warnings.append("В текстовом слое PDF не подтверждён номер: нужна визуальная проверка")
    if not year:
        warnings.append("В текстовом слое PDF не подтверждён год: нужна визуальная проверка")
    if not labels:
        warnings.append("В текстовом слое PDF не найдены основные метки; определить по OCR")
    for section in ("", "B", "C"):
        numbers = [int(label[len(section):]) for label in labels if re.fullmatch(section + r"\d+", label)]
        if numbers and sorted(numbers) != list(range(1, max(numbers) + 1)):
            warnings.append(f"В текстовом слое пропуски меток части {section or 'числовой'}: {sorted(numbers)}; проверить по изображению")
    return {"page_count": len(texts), "page_labels": page_labels, "source_labels": labels,
            "page_positions": page_positions, "page_dimensions": page_dimensions,
            "pdf_text": text, "warnings": warnings}


def validate_image(data):
    """Проверить сигнатуру и декодирование PNG/JPEG, вернув исходные байты."""
    if not (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff")):
        raise ValueError("Файл не похож на PNG/JPEG опубликованных ответов")
    try:
        pixmap = pymupdf.Pixmap(data)
        if pixmap.width <= 0 or pixmap.height <= 0:
            raise ValueError("Пустое изображение")
    except (RuntimeError, ValueError) as error:
        raise ValueError("Не удалось декодировать изображение ответов") from error
    return data


def parse_script_answers(data):
    """Распаковать JS и прочитать опубликованные ответы T<n> строками.

    Число ответов не фиксируется: оно зависит от формата варианта.
    Пустой разбор и повторяющиеся метки отклоняются без угадывания ответов.
    """
    try:
        text = data.decode("utf-16") if data.startswith((b"\xff\xfe", b"\xfe\xff")) else data.decode("utf-8-sig")
        if packer.detect(text):
            text = packer.unpack(text)
        text = jsbeautifier.beautify(text)
        matches = re.findall(r'getElementById\(\s*["\']T(\d+)["\']\s*\)\.value\s*=\s*["\']([^"\']+)["\']', text)
    except (UnicodeError, RuntimeError, ValueError, IndexError, UnpackingError) as error:
        raise ValueError("Невозможно разобрать JS ответов") from error
    if not matches:
        raise ValueError("В JS нет опубликованных ответов")
    result = {}
    for label, value in matches:
        label = normalize_label(label)
        if label in result:
            raise ValueError(f"Повтор ответа {label} в JS")
        result[label] = value
    return result
