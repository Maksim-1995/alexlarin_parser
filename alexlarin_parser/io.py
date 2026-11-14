"""Общие операции записи, чтения JSON и вычисления хешей без сетевых вызовов.

Запись через временный файл сохраняет прежний источник или кэш до полной
готовности новых данных. Формат отпечатков используется в идентичности кэшей.
"""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def json_value(value):
    """Рекурсивно преобразовать Path в строки для проверочных JSON-отчётов."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def fingerprint(value):
    """Получить SHA-256 JSON-значения с сортировкой ключей и стабильным форматом.

    Разделители и параметры сериализации входят в контракт кэшей: менять их
    при рефакторинге нельзя, иначе прежние отпечатки перестанут совпадать.
    """
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def file_sha256(path):
    """Вернуть SHA-256 фактических байтов файла, независимо от его имени."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_bytes(path, data):
    """Записать байты целиком, затем атомарно заменить файл назначения.

    Временный файл создаётся рядом с целевым, чтобы замена происходила
    в одной файловой системе. При ошибке записи прежний файл сохраняется,
    а временный удаляется. Каталог назначения создаётся при необходимости.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as target:
            pending = Path(target.name)
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        pending.replace(path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def atomic_json(path, value, *, trailing_newline=False):
    """Атомарно сохранить JSON в UTF-8, сохранив принятый формат отступов.

    ``trailing_newline`` нужен манифестам исходников, исторически завершающимся
    переводом строки. По умолчанию формат существующих AI-кэшей не меняется.
    """
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if trailing_newline:
        text += "\n"
    atomic_bytes(path, text.encode("utf-8"))


def read_json(path):
    """Прочитать JSON или вернуть None для отсутствующего/нечитаемого кэша.

    Проверку структуры и совместимости прочитанных данных выполняет
    вызывающий этап; успешный разбор JSON сам по себе не подтверждает кэш.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
