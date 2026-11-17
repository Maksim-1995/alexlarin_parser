"""Проверенные ручные категории поверх исходного AI-кэша одного варианта."""

import json
from pathlib import Path

from .classification.catalogues import load_catalog, validate_classification
from .io import file_sha256


DOCUMENT_FIELDS = {"version", "variant_key", "source_url", "pdf_sha256", "corrections"}
RECORD_FIELDS = {"source_label", "field", "old", "new", "reason", "evidence", "method"}
CATEGORY_FIELDS = {"exams_id", "topics_id"}


def apply_category_corrections(context, tasks, *, pdf_path, exams_path, topics_path):
    """Проверить все правки, затем применить их к задачам, не записывая AI-кэш.

    Отсутствие файла сохраняет обычный поток. Неверная identity, устаревшее
    old, повторная/неизвестная метка или недопустимая категория вызывают
    ValueError до изменения задач. Возвращается provenance для отчёта и сверки.
    Семантическое основание каждой правки указывает автор JSON, не валидатор.
    """
    path = Path(context.base_dir) / "harness" / "corrections" / f"{context.variant_key}.json"
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError("Коррекции: файл должен содержать UTF-8 JSON") from error
    if not isinstance(document, dict) or set(document) != DOCUMENT_FIELDS:
        raise ValueError("Коррекции: неверные поля документа")
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("Коррекции: неподдерживаемая версия")
    if (document["variant_key"] != context.variant_key
            or document["source_url"] != context.source_url
            or document["pdf_sha256"] != file_sha256(pdf_path)):
        raise ValueError("Коррекции: вариант, URL или SHA-256 PDF не совпадают")
    records = document["corrections"]
    if not isinstance(records, list) or not records:
        raise ValueError("Коррекции: нужен непустой список правок")
    prepared = {task["source_label"]: dict(task) for task in tasks}
    if len(prepared) != len(tasks):
        raise ValueError("Коррекции: исходные метки задач повторяются")
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
            raise ValueError("Коррекции: неверные поля записи")
        label, field = record["source_label"], record["field"]
        if not isinstance(label, str) or label not in prepared:
            raise ValueError("Коррекции: неизвестная исходная метка")
        if not isinstance(field, str) or field not in CATEGORY_FIELDS:
            raise ValueError("Коррекции: разрешены только exams_id и topics_id")
        if (label, field) in seen:
            raise ValueError("Коррекции: повторная правка одного поля задачи")
        seen.add((label, field))
        if any(not isinstance(record[name], str) or not record[name].strip()
               for name in ("reason", "method")):
            raise ValueError("Коррекции: нужны причина и метод проверки")
        evidence = record["evidence"]
        if (not isinstance(evidence, list) or not evidence
                or any(not isinstance(item, str) or not item.strip() for item in evidence)):
            raise ValueError("Коррекции: нужны доказательства проверки")
        current = prepared[label][field]
        if type(current) is not type(record["old"]) or current != record["old"]:
            raise ValueError(f"Коррекции: прежнее значение {label}.{field} изменилось")
        prepared[label][field] = record["new"]

    exams = load_catalog(exams_path, name_column="name_of_Exam",
                         parent_column="parent_of_exam", root_id=2)
    topics = load_catalog(topics_path, name_column="name", parent_column="parent", root_id=1)
    validated = validate_classification(
        [{"task_num": task["task_num"], **{field: task[field] for field in CATEGORY_FIELDS}}
         for task in prepared.values()],
        exams, topics, task_numbers={task["task_num"] for task in tasks},
    )
    for task in prepared.values():
        for field in CATEGORY_FIELDS:
            value = validated[task["task_num"]][field]
            if type(value) is not type(task[field]) or value != task[field]:
                raise ValueError("Коррекции: укажите существующий ID или точное новое название")
    # Применяем только после проверки всех записей и полного набора категорий.
    for task in tasks:
        task.update({field: prepared[task["source_label"]][field] for field in CATEGORY_FIELDS})
    return {"path": str(path), "sha256": file_sha256(path), "records": records}
