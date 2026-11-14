"""Совместимые прежние импорты источников; реализация находится в пакете.

Контекст принадлежит alexlarin_parser.context, чистый разбор — sources.parsing,
каталог ссылок — sources.catalogue, подготовка файлов — sources.service.
Подмены зависимостей в тестах следует направлять в модуль-владелец функции.
"""

from alexlarin_parser.context import CODE_RE, VariantContext, _canonical_url
from alexlarin_parser.sources import (
    ARCHIVE_URL,
    LEGACY_URL,
    LABEL_RE,
    SourceHTMLParser,
    atomic_bytes,
    atomic_json,
    decode_html,
    discover_catalog,
    download_ranged_source,
    inspect_pdf,
    load_catalogue,
    normalize_label,
    parse_script_answers,
    parse_variant_html,
    prepare_sources,
    select_variants,
    task_positions,
    validate_image,
)
# Частные прежние helpers сохраняются для существующих локальных инструментов.
from alexlarin_parser.sources.service import _load, _manifest


__all__ = [
    "ARCHIVE_URL", "CODE_RE", "LEGACY_URL", "LABEL_RE", "SourceHTMLParser",
    "VariantContext", "atomic_bytes", "atomic_json", "decode_html",
    "discover_catalog", "download_ranged_source", "inspect_pdf", "load_catalogue",
    "normalize_label", "parse_script_answers", "parse_variant_html",
    "prepare_sources", "select_variants", "task_positions", "validate_image",
]
