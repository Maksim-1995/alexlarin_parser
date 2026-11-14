"""Совместимый импорт; реализация находится в alexlarin_parser.sources.downloader."""

import sys
from alexlarin_parser.sources import downloader as _implementation

sys.modules[__name__] = _implementation
