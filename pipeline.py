"""Совместимый импорт; реализация находится в alexlarin_parser.pipeline."""

import sys
from alexlarin_parser import pipeline as _implementation

sys.modules[__name__] = _implementation
