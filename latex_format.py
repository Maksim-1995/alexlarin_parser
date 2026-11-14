"""Совместимый импорт; реализация находится в alexlarin_parser.formatting.latex."""

import sys
from alexlarin_parser.formatting import latex as _implementation

sys.modules[__name__] = _implementation
