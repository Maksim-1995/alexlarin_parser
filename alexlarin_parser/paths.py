"""Пути проекта, не зависящие от текущего рабочего каталога процесса."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
