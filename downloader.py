"""Загрузка содержимого файлов по HTTP."""

import requests


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
