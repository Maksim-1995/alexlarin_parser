"""Подготовка изображений страницы без модельных запросов."""

import base64

from ..sources.parsing import task_positions


def _image_part(data):
    """Упаковать PNG в image_url-часть запроса без записи промежуточного файла."""
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
        "detail": "original",
    }}


def render_scheme(page, dpi):
    """Выбрать полную страницу или две колонки по координатам начал задач.

    Возвращаемые области и параметры рендеринга входят в identity OCR-кэша.
    Колонки используются только при двух раздельных группах меток с достаточным
    зазором; обе области затем передаются в одном запросе для этой страницы.
    """
    positions = task_positions(page)
    width, height = page.rect.width, page.rect.height
    starts = sorted({round(p["x"], 1) for p in positions})
    left = [x for x in starts if x < width * .35]
    right = [x for x in starts if width * .48 < x < width * .8]
    split = bool(left and right and min(right) - max(left) > width * .25)
    if split:
        clips = [[0, 0, width / 2 + 3, height], [width / 2 - 3, 0, width, height]]
    else:
        clips = [[0, 0, width, height]]
    return {"version": 2, "dpi": dpi, "colorspace": "RGB", "alpha": False,
            "layout": "two_columns" if split else "full_page", "clips": clips}


