"""Совместимый импорт; реализация находится в alexlarin_parser.io."""

import sys
from alexlarin_parser import io as _implementation

sys.modules[__name__] = _implementation
