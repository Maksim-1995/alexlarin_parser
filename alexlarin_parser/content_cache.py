"""Кэш проверенных ответов по входу модели, без идентификаторов варианта."""

from pathlib import Path

from .io import atomic_json, fingerprint, read_json
from .model_response import ModelResponse


def response_cache_path(context, stage, identity, *, cache_dir=None):
    """Обычные запуски делят кэш; явный cache_dir изолирует fresh/resume.

    В файле нет запроса, base64 или ключей. identity содержит только хеш
    сообщений и публичные настройки совместимости, подготовленные этапом.
    """
    root = (Path(cache_dir) / "by_content" if cache_dir is not None else
            Path(context.base_dir) / "input" / "cache" / "model_responses")
    return root / stage / f"{fingerprint(identity)}.json"


def read_response(path, identity, validate):
    """Проверить identity, целостность ответа и схему до использования кэша."""
    cached = read_json(path)
    try:
        if not isinstance(cached, dict) or cached.get("identity") != identity:
            return None
        response = cached["response"]
        if cached.get("response_sha256") != fingerprint(response):
            return None
        if "record_sha256" in cached and cached["record_sha256"] != fingerprint({
            "response": response, "provenance": cached.get("provenance"),
        }):
            return None
        validate(response)
        return ModelResponse(response, cached.get("provenance"))
    except (KeyError, TypeError, ValueError):
        return None


def write_response(path, identity, response):
    """Записать ответ атомарно; вызывающий этап уже обязан проверить схему."""
    record = {"response": response, "provenance": getattr(response, "provenance", None)}
    atomic_json(path, {"identity": identity, "response_sha256": fingerprint(response),
                       "record_sha256": fingerprint(record), **record})
