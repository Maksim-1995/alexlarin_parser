"""Совместимый импорт; реализация находится в alexlarin_parser.formatting.math_text."""

import sys
from alexlarin_parser.formatting import math_text as _implementation

sys.modules[__name__] = _implementation
