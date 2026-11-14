"""Совместимый запуск CLI; реализация находится в alexlarin_parser.cli."""

import sys
from alexlarin_parser import cli as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.cli())

sys.modules[__name__] = _implementation
