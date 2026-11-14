"""Реальные ссылки архива, локальный каталог и выбор основных вариантов.

Адреса вариантов берутся из href; дополнительные версии остаются отдельными
записями. Проверка конкретных HTML/PDF и привязка файлов находятся в service.
"""

from hashlib import sha256
from pathlib import Path
from urllib.parse import urljoin

from ..context import VariantContext, _canonical_url
from ..io import atomic_bytes, atomic_json
from . import downloader
from .parsing import SourceHTMLParser, decode_html


ARCHIVE_URL = "https://alexlarin.net/trvar.html"


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
    """Прочитать сохранённый каталог href или скачать и проверить недостающий.

    В режиме ``local_only`` сеть запрещена; прежний файл заменяется только
    после успешного разбора нового HTML и обнаружения ссылок вариантов.
    """
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
    atomic_json(path.with_suffix(".json"), {"source_url": ARCHIVE_URL, "sha256": sha256(data).hexdigest(), "variants": catalog}, trailing_newline=True)
    return catalog


def select_variants(numbers, catalogue):
    """Выбрать основные варианты в заданном порядке без подмены версиями 4*.

    Отсутствующие и неоднозначные ссылки отклоняются до обработки вариантов.
    """
    selected = []
    for number in numbers:
        choices = [entry for entry in catalogue if entry["number"] == int(number) and entry["code"] == f"trvar{int(number)}"]
        if len(choices) != 1:
            raise ValueError(f"Основной вариант №{number} не найден однозначно: {len(choices)} ссылок")
        selected.append(choices[0])
    return selected
