"""Совместимый импорт; реализация находится в alexlarin_parser.ocr.recognition."""

import sys
from alexlarin_parser.ocr import recognition as _implementation

sys.modules[__name__] = _implementation
