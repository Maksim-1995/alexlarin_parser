"""Совместимые импорты изолированной выдачи; владельцы — exporting-модули.

Старые публичные имена и используемые локальными проверками helpers доступны
через обычные импорты. Подменять зависимости тестам следует у их владельца.
"""

from alexlarin_parser.exporting.archive import create_zip
from alexlarin_parser.exporting.images import (
    _column_starts,
    _image_placement_identity,
    _owner,
    _page_positions,
    _render_image_placement,
    _vector_rects,
    extract_variant_images,
)
from alexlarin_parser.exporting.metadata import ABOUT_FIELDS, prepare_about_metadata
from alexlarin_parser.exporting.service import (
    IMAGE_EXTENSIONS,
    TASK_FIELDS,
    _publish,
    _validate_result,
    export_variant,
    format_condition_tables,
    normalize_answer,
)
from alexlarin_parser.exporting.workbook import write_excel

__all__ = [
    "ABOUT_FIELDS", "IMAGE_EXTENSIONS", "TASK_FIELDS", "create_zip",
    "export_variant", "extract_variant_images", "format_condition_tables",
    "normalize_answer", "prepare_about_metadata", "write_excel",
]
