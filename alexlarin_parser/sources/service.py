"""Подготовка проверенных источников одного варианта и их манифеста.

Модуль переиспользует URL/SHA-256-привязки, загружает недостающие файлы,
считает HTTP-попытки и сохраняет только проверенные результаты. Чистый
разбор содержимого расположен в parsing, общий атомарный IO — в io.
"""

from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit

import requests

from ..context import _canonical_url
from ..io import atomic_bytes, atomic_json as _atomic_json
from . import downloader
from .parsing import (
    inspect_pdf, normalize_label, parse_script_answers, parse_variant_html, validate_image,
)


LEGACY_URL = "https://alexlarin.net/ege/2026/trvar542.html"


def atomic_json(path, data):
    """Сохранить JSON общим писателем с прежним переводом строки в конце.

    Имя реэкспортируется через variant_sources для прежних вызовов.
    Подмены в тестах адресуются этому модулю; запись реализована в
    alexlarin_parser.io.
    """
    _atomic_json(path, data, trailing_newline=True)


def _manifest(context):
    """Прочитать привязки файлов к URL/хешам только для текущего варианта."""
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
    """Вернуть проверенный источник, сохранив прежний файл при сбое загрузки.

    Бинарный файл требует совпадения URL и SHA-256 в манифесте. Для HTML/PDF
    ``semantic_identity`` допускает первичную привязку по содержимому; уже
    существующая несовпадающая запись манифеста этим правилом не обходится.
    Локальная миграция разрешена только по точному URL известного варианта 542.
    """
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
    """Подготовить проверенные источники и метаданные одного варианта без AI.

    Возвращает PDF, опубликованные ответы, изображения ответов, основные метки
    и геометрию страниц. Ошибка обязательных HTML/PDF останавливает этап;
    недоступный дополнительный источник ответов попадает в ``warnings``,
    чтобы сохранить условия и явно отметить неполноту результата.
    """
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
