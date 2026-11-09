"""Идентичность варианта, реальный каталог архива и проверенные исходники.

В этом модуле нет обращений к моделям. Любой локальный бинарный источник
связан с точным исходным URL и SHA-256 в source_manifest.json.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import jsbeautifier
from jsbeautifier.unpackers import packer, UnpackingError
import pymupdf
import requests

import downloader


ARCHIVE_URL = "https://alexlarin.net/trvar.html"
LEGACY_URL = "https://alexlarin.net/ege/2026/trvar542.html"
LABEL_RE = re.compile(r"^([BВCС]?)[ \t]*(\d+)[.)]?$", re.I)
CODE_RE = re.compile(r"^trvar(?P<number>\d+)(?P<suffix>[\w*+.-]*)$", re.ASCII)


def normalize_label(label):
    """B/В и C/С — явные визуальные эквиваленты, остальные буквы запрещены."""
    if not isinstance(label, str):
        raise ValueError(f"Метка задачи должна быть строкой: {label!r}")
    match = LABEL_RE.fullmatch(label.strip())
    if not match or int(match[2]) <= 0:
        raise ValueError(f"Некорректная метка основной задачи: {label!r}")
    letter = match[1].upper().translate(str.maketrans({"В": "B", "С": "C"}))
    return f"{letter}{int(match[2])}"


def _canonical_url(url):
    parts = urlsplit(unescape(url))
    if parts.scheme not in {"http", "https"} or parts.hostname not in {"alexlarin.net", "www.alexlarin.net"}:
        raise ValueError(f"Нужна ссылка на alexlarin.net: {url}")
    if parts.username or parts.password or parts.port not in {None, 80, 443}:
        raise ValueError("Недопустимые параметры адреса источника")
    return urlunsplit(("https", "alexlarin.net", parts.path, parts.query, ""))


@dataclass
class VariantContext:
    variant_key: str
    number: int
    code: str
    year: int
    source_url: str
    base_dir: Path
    source_urls: dict = field(default_factory=dict)
    download_count: int = 0

    @classmethod
    def from_url(cls, url, base_dir):
        url = _canonical_url(url)
        path = Path(unquote(urlsplit(url).path))
        match = CODE_RE.fullmatch(path.stem)
        if path.suffix.lower() != ".html" or not match or not path.parent.name.isdigit():
            raise ValueError(f"Ссылка не описывает вариант ЕГЭ: {url}")
        year = int(path.parent.name)
        if not 2000 <= year <= 2100:
            raise ValueError(f"Некорректный год варианта: {year}")
        # ~ не допускается в исходном code: экранирование не сталкивается
        # с реально существующими суффиксами вроде _star или _plus.
        safe_code = path.stem.replace("*", "~2a").replace("+", "~2b")
        key = f"ege_{year}_{safe_code}"
        if urlsplit(url).query:
            key += "~q" + sha256(urlsplit(url).query.encode()).hexdigest()[:12]
        return cls(key, int(match["number"]), path.stem, year, url, Path(base_dir))

    @property
    def variant_dir(self):
        return self.base_dir / "input" / "variants" / self.variant_key

    @property
    def source_dir(self):
        return self.variant_dir / "source"

    @property
    def cache_dir(self):
        return self.variant_dir / "cache"

    @property
    def debug_dir(self):
        return self.variant_dir / "debug"

    @property
    def output_dir(self):
        return self.base_dir / "output" / self.variant_key

    @property
    def archive_path(self):
        return self.base_dir / "archives" / f"{self.variant_key}_result.zip"

    @property
    def pdf_path(self):
        return self.source_dir / f"{self.code}.pdf"

    @property
    def html_path(self):
        return self.source_dir / f"{self.code}.html"


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as out:
            pending = Path(out.name)
            out.write(data)
        pending.replace(path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf8"))


def decode_html(data):
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
        return " ".join("".join(self.title_parts).split())


def parse_variant_html(data, context):
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


def discover_catalog(data, archive_url=ARCHIVE_URL):
    """Вернуть реальные href; дополнительные версии остаются отдельными записями."""
    parser = SourceHTMLParser()
    parser.feed(decode_html(data) if isinstance(data, bytes) else data)
    urls = []
    for tag, href in parser.links:
        if tag != "a":
            continue
        try:
            url = _canonical_url(urljoin(archive_url, href))
            context = VariantContext.from_url(url, Path("."))
        except ValueError:
            continue
        if url not in urls:
            urls.append(url)
    if not urls:
        raise ValueError("В HTML архива не найдены ссылки вариантов")
    return [{"variant_key": c.variant_key, "number": c.number, "code": c.code,
             "year": c.year, "source_url": c.source_url}
            for c in (VariantContext.from_url(url, Path(".")) for url in urls)]


def load_catalogue(base_dir, *, local_only=False):
    path = Path(base_dir) / "input" / "catalogue" / "trvar.html"
    if path.is_file():
        try:
            return discover_catalog(path.read_bytes())
        except ValueError:
            if local_only:
                raise
    if local_only:
        raise FileNotFoundError(f"Нет проверенного локального каталога: {path}")
    data = downloader.download_bytes(ARCHIVE_URL)
    catalog = discover_catalog(data)
    atomic_bytes(path, data)
    atomic_json(path.with_suffix(".json"), {"source_url": ARCHIVE_URL, "sha256": sha256(data).hexdigest(), "variants": catalog})
    return catalog


def select_variants(numbers, catalogue):
    selected = []
    for number in numbers:
        choices = [entry for entry in catalogue if entry["number"] == int(number) and entry["code"] == f"trvar{int(number)}"]
        if len(choices) != 1:
            raise ValueError(f"Основной вариант №{number} не найден однозначно: {len(choices)} ссылок")
        selected.append(choices[0])
    return selected


def task_positions(page):
    """Основные метки PDF; подпункты А), Б), В) без числа не подходят."""
    positions = []
    words = page.get_text("words")
    visible_words = [pymupdf.Rect(*word[:4]) * page.rotation_matrix for word in words]
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
            year = re.search(r"(?:экзамен[,\s]+|ЕГЭ\s*[-–]?\s*)(20\d\d)(?:\s*[-–]\s*(20\d\d))?", text, re.I)
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


def _manifest(context):
    path = context.source_dir / "source_manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf8"))
    except (FileNotFoundError, ValueError, UnicodeError):
        return {"variant_key": context.variant_key, "source_url": context.source_url, "files": {}}
    if data.get("variant_key") != context.variant_key or data.get("source_url") != context.source_url:
        raise ValueError("Манифест исходников относится к другому варианту")
    return data


def download_ranged_source(context, url, *, chunk_size=8192):
    """Собрать источник из проверенных HTTP Range-ответов при зависании GET.

    Некоторые старые PDF сайта перестают передаваться после первых 12 KiB.
    Короткие диапазоны обходим без повторной оплаты модели. Каждый HTTP
    запрос учитывается отдельно; произвольные пропуски и перекрытия запрещены.
    Частичный файл никогда не устанавливается в качестве исходника.
    """
    chunks = []
    offset = 0
    total = None
    etag = None
    while total is None or offset < total:
        end = offset + chunk_size - 1 if total is None else min(total - 1, offset + chunk_size - 1)
        headers = {"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"}
        if etag:
            headers["If-Range"] = etag
        context.download_count += 1
        with requests.get(url, headers=headers, timeout=(10, 30)) as response:
            response.raise_for_status()
            data = response.content
            if response.status_code == 200 and offset == 0:
                if not data:
                    raise ValueError("Пустой ответ на Range-запрос")
                return data
            if response.status_code != 206:
                raise ValueError(f"Ожидался HTTP206 для диапазона {offset}-{end}: {response.status_code}")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match:
                raise ValueError("Нет корректного Content-Range в ответе источника")
            start, actual_end, current_total = map(int, match.groups())
            if not 0 < current_total <= 100 * 1024 * 1024:
                raise ValueError("Некорректный полный размер Range-источника")
            expected_end = min(end, current_total - 1)
            if start != offset or actual_end != expected_end or len(data) != actual_end - start + 1:
                raise ValueError(f"Неполный или смещённый HTTP Range {start}-{actual_end}")
            if total is not None and current_total != total:
                raise ValueError("Полный размер источника изменился между Range-запросами")
            current_etag = response.headers.get("ETag")
            if etag is not None and current_etag != etag:
                raise ValueError("ETag источника изменился между Range-запросами")
            etag = current_etag
            total = current_total
            chunks.append(data)
            offset = actual_end + 1
    result = b"".join(chunks)
    if len(result) != total:
        raise ValueError("Размер собранного Range-источника не совпадает с заголовком")
    return result


def _load(context, url, path, validate, *, local_only=False, semantic_identity=False):
    """Проверенный source cache; никакого поиска подходящего файла соседнего варианта."""
    manifest = _manifest(context)
    record = manifest.get("files", {}).get(path.name)
    local_error = None
    if path.is_file():
        data = path.read_bytes()
        bound = (record and record.get("url") == url and record.get("sha256") == sha256(data).hexdigest())
        try:
            if not bound and not semantic_identity:
                raise ValueError("Локальный бинарный источник не связан с URL/хешем в манифесте")
            result = validate(data)
            if bound:
                return result
            # HTML/PDF можно проверить по содержимому и связать без сети.
            if not record:
                manifest.setdefault("files", {})[path.name] = {"url": url, "sha256": sha256(data).hexdigest(), "method": "validated_content"}
                atomic_json(context.source_dir / "source_manifest.json", manifest)
                return result
        except ValueError as error:
            local_error = error
    # Единственная разрешённая миграция старых плоских исходников.
    if context.source_url == LEGACY_URL and context.code == "trvar542":
        legacy = context.base_dir / "input" / path.name
        expected = _canonical_url(urljoin(LEGACY_URL, path.name))
        if legacy.is_file() and url == expected:
            data = legacy.read_bytes()
            result = validate(data)
            atomic_bytes(path, data)
            manifest.setdefault("files", {})[path.name] = {"url": url, "sha256": sha256(data).hexdigest(), "method": "explicit_542_legacy_migration", "legacy_path": str(legacy)}
            atomic_json(context.source_dir / "source_manifest.json", manifest)
            return result
    if local_only:
        raise FileNotFoundError(f"Нет проверенного локального источника {url}: {local_error or path}")
    context.download_count += 1
    try:
        try:
            data = downloader.download_bytes(url)
        except (requests.ReadTimeout, requests.ConnectionError):
            # Единственная автоматическая смена транспорта касается исходников,
            # не модельных запросов. Для HTML сохраняется обычная ошибка.
            if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
                raise
            data = download_ranged_source(context, url)
    except requests.RequestException as error:
        raise RuntimeError(f"Не удалось загрузить источник {url}: {error}") from error
    if not data:
        raise ValueError(f"Пустой источник {url}")
    result = validate(data)
    atomic_bytes(path, data)
    manifest.setdefault("files", {})[path.name] = {"url": url, "sha256": sha256(data).hexdigest(), "method": "http_download"}
    atomic_json(context.source_dir / "source_manifest.json", manifest)
    return result


def prepare_sources(context, *, local_only=False):
    """Скачать только недостающие источники конкретного варианта без AI."""
    html = _load(context, context.source_url, context.html_path,
                 lambda data: parse_variant_html(data, context), local_only=local_only, semantic_identity=True)
    resources = html["resources"]
    context.source_urls.update(html=context.source_url, pdf=resources["pdf"][0],
                               js=resources["js"], answer_images=resources["images"])
    pdf = _load(context, resources["pdf"][0], context.pdf_path,
                lambda data: inspect_pdf(data, context), local_only=local_only, semantic_identity=True)
    answers = dict(html["answers"])
    warnings = list(pdf["warnings"])
    for url in resources["js"]:
        path = context.source_dir / Path(urlsplit(url).path).name
        try:
            script_answers = _load(context, url, path, parse_script_answers, local_only=local_only)
            for label, answer in script_answers.items():
                if label in answers and answers[label] != answer:
                    raise ValueError(f"Конфликт HTML/JS ответов {label}")
                answers[label] = answer
        except (ValueError, RuntimeError, FileNotFoundError) as error:
            warnings.append(str(error))
    images = []
    for url in resources["images"]:
        path = context.source_dir / Path(urlsplit(url).path).name
        try:
            _load(context, url, path, validate_image, local_only=local_only)
            images.append(path)
        except (ValueError, RuntimeError, FileNotFoundError) as error:
            warnings.append(str(error))
    extra_answers = {}
    if context.source_url == LEGACY_URL and resources["images"]:
        manual_path = context.source_dir / "trvar542_answers.json"
        if not manual_path.is_file():
            manual_path = context.base_dir / "input" / "trvar542_answers.json"
        if manual_path.is_file():
            extra = json.loads(manual_path.read_text(encoding="utf8"))
            if extra.get("source_url") == resources["images"][0] and extra.get("method") and isinstance(extra.get("answers"), dict):
                for label, value in extra["answers"].items():
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError("Неполный ручной перенос ответов 542")
                    extra_answers[normalize_label(label)] = value
                atomic_json(context.source_dir / "trvar542_answers.json", extra)
            else:
                warnings.append("Ручные ответы 542 не соответствуют опубликованному источнику")
    metadata = {"class": 11, "year": context.year, "date": None, "title": f"Тренировочный вариант № {context.number}",
                "code": context.code, "source_name": "alexlarin.net", "source_url": context.source_url}
    result = {**pdf, "pdf_path": context.pdf_path, "html_answers": answers,
              "original_answer_labels": html["answer_labels"], "answer_images": images,
              "extra_answers": extra_answers, "metadata": metadata, "warnings": warnings,
              "download_count": context.download_count, "source_urls": context.source_urls}
    atomic_json(context.debug_dir / "source_inspection.json",
                {key: value for key, value in result.items() if key not in {"pdf_path", "answer_images", "pdf_text"}})
    return result
