"""Совместимый импорт; реализация находится в alexlarin_parser.requests_counter."""

import sys
from alexlarin_parser import requests_counter as _implementation

sys.modules[__name__] = _implementation
