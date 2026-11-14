"""Совместимый импорт; реализация находится в alexlarin_parser.provider."""

import sys
from alexlarin_parser import provider as _implementation

sys.modules[__name__] = _implementation
