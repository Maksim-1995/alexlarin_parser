"""Загрузка версионированных текстов промптов без изменения их пробелов."""

from importlib.resources import files


def load_prompt(name):
    """Прочитать UTF-8 ресурс целиком: переносы входят в identity AI-кэша."""
    if name not in {
        "ocr.txt", "classification.txt", "classification_numeric_legacy.txt",
        "classification_gemini_legacy.txt",
    }:
        raise ValueError(f"Неизвестный промпт: {name}")
    return files(__package__).joinpath(name).read_bytes().decode("utf-8")
