"""Небольшие общие операции записи и идентичности без сетевых эффектов."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as target:
            pending = Path(target.name)
            json.dump(value, target, ensure_ascii=False, indent=2)
            target.flush()
            os.fsync(target.fileno())
        pending.replace(path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
