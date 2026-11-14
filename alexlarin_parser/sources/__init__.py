"""Каталог архива, разбор и подготовка изолированных источников вариантов."""

from ..context import VariantContext
from ..io import atomic_bytes
from .catalogue import ARCHIVE_URL, discover_catalog, load_catalogue, select_variants
from .parsing import (
    LABEL_RE,
    SourceHTMLParser,
    decode_html,
    inspect_pdf,
    normalize_label,
    parse_script_answers,
    parse_variant_html,
    task_positions,
    validate_image,
)
from .service import LEGACY_URL, atomic_json, download_ranged_source, prepare_sources


__all__ = [
    "ARCHIVE_URL", "LEGACY_URL", "LABEL_RE", "SourceHTMLParser", "VariantContext",
    "atomic_bytes", "atomic_json", "decode_html", "discover_catalog",
    "download_ranged_source", "inspect_pdf", "load_catalogue", "normalize_label",
    "parse_script_answers", "parse_variant_html", "prepare_sources",
    "select_variants", "task_positions", "validate_image",
]
