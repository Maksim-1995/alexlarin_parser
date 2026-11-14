"""Совместимый импорт; реализация находится в alexlarin_parser.formatting.html."""

import sys
from alexlarin_parser.formatting import html as _implementation

sys.modules[__name__] = _implementation
