"""Подготовка полей листа about по шаблону работодателя."""

from urllib.parse import urlsplit


ABOUT_FIELDS = (
    "class",
    "year",
    "date",
    "topic",
    "exam_id",
    "title",
    "code",
    "source_name",
    "is_public",
    "source_url",
    "description",
)


def prepare_about_metadata(metadata):
    """Вернуть поля для Excel, сохранив исходный словарь без изменений.

    Неизвестные значения остаются пустыми. Поле date сохраняет прежнее
    значение и смысл: модуль не подменяет дату выгрузки датой публикации.
    """
    result = {field: metadata.get(field) for field in ABOUT_FIELDS}
    result["title"] = metadata.get("title") or metadata.get("variant_name")

    if not result["source_name"] and result["source_url"]:
        result["source_name"] = urlsplit(result["source_url"]).hostname

    return result
