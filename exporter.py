"""Совместимые импорты Excel/ZIP; реализация находится в exporting-пакете."""

from alexlarin_parser.exporting.archive import create_zip
from alexlarin_parser.exporting.workbook import format_excel, write_excel

__all__ = ["create_zip", "format_excel", "write_excel"]
