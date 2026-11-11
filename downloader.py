"""Загрузка исходников с проверкой и повторным использованием локальных файлов."""

from pathlib import Path

import requests

from pipeline_io import atomic_bytes


def download_bytes(url, timeout=30):
    """Скачать содержимое ресурса в виде байтов.

    Args:
        url (str): Адрес ресурса.
        timeout (int | float | tuple): Тайм-аут в секундах для
            подключения и чтения либо пара отдельных тайм-аутов.

    Returns:
        bytes: Содержимое успешного HTTP-ответа.

    Raises:
        requests.RequestException: При ошибке запроса, включая
            превышение тайм-аута и ошибочный статус HTTP-ответа.
    """
    with requests.get(url, timeout=timeout) as response:
        response.raise_for_status()
        return response.content


def load_source(url, path, validate):
    """Вернуть проверенный локальный исходник или скачать недостающий.

    ``validate`` разбирает байты и возвращает нужные вызывающему данные.
    Непригодный исходник отклоняется через ValueError. Новый файл заменяет
    прежний только после проверки, поэтому страница ошибки не попадёт в кэш.

    Это базовый загрузчик для сохранённых плоских исходников. Связь файла
    с вариантом, URL и хешем дополнительно проверяет variant_sources.
    """
    path = Path(path)
    local_error = None
    if path.is_file():
        try:
            data = path.read_bytes()
            if not data:
                raise ValueError("пустой файл")
            result = validate(data)
        except ValueError as error:
            local_error = error
            print(f"Локальный исходник не подходит: {path}: {error}", flush=True)
        else:
            print(f"Используем сохранённый исходник: {path}", flush=True)
            return result

    try:
        data = download_bytes(url)
    except requests.RequestException as error:
        reason = f" ({local_error})" if local_error else ""
        raise RuntimeError(
            f"Нет подходящего локального исходника {path}{reason}; "
            f"не удалось загрузить {url}: {error}"
        ) from error
    try:
        if not data:
            raise ValueError("пустой ответ сервера")
        result = validate(data)
    except ValueError as error:
        raise ValueError(f"Непригодный исходник {url}: {error}") from error

    atomic_bytes(path, data)
    print(f"Исходник сохранён: {path}", flush=True)
    return result
