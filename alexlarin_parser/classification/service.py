"""Классификация одного изолированного варианта общим DeepSeek-клиентом.

Сервис готовит E/T-коды допустимых листьев, проверяет ответ и управляет
точным кэшем. Его правила совместимости включают предыдущий числовой
DeepSeek-протокол; Gemini-кэш сюда не переносится.
"""

import hashlib
import json
from pathlib import Path
import re

from alexlarin_parser.io import atomic_json, fingerprint as _variant_fingerprint
from alexlarin_parser.prompts import load_prompt
from .catalogues import allowed_categories, load_catalog, validate_classification


VARIANT_RULES_VERSION = "variant-math-descendant-leaves-new-topics-v1"
# Прежний числовой протокол нужен только для точной проверки старого кэша.
NUMERIC_VARIANT_PROMPT_VERSION = 'deepseek-numeric-leaves-v2-independent-whitelists'
NUMERIC_VARIANT_PROMPT = load_prompt("classification_numeric_legacy.txt")
NUMERIC_VARIANT_SCHEMA = {
    "type": "object",
    "required": ["tasks"],
    "additionalProperties": False,
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["task_num", "exams_id", "topics_id"],
                "additionalProperties": False,
                "properties": {
                    "task_num": {"type": "integer"},
                    "exams_id": {"type": "integer"},
                    "topics_id": {"type": ["integer", "string"]},
                },
            },
        },
    },
}

VARIANT_CACHE_VERSION = 2
VARIANT_PROMPT_VERSION = "deepseek-namespaced-leaves-v3"
VARIANT_PROMPT = load_prompt("classification.txt")
VARIANT_SCHEMA = {
    **NUMERIC_VARIANT_SCHEMA,
    "model_choices": {"exams_id": "E label", "topics_id": "T label or new topic name"},
}


def prepare_variant_tasks(tasks):
    """Вернуть отсортированные копии task_num/condition/source_label.

    Нужен непустой список с номерами 1..N, уникальными исходными метками
    и непустыми условиями. Неверный вход вызывает ValueError до модели.
    Ответы, рисунки и технические поля не передаются в классификацию;
    исходные словари задач не изменяются.
    """
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Нужен непустой список задач одного варианта")
    prepared = []
    labels = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Каждое задание должно быть словарём")
        number, condition = task.get("task_num"), task.get("condition")
        if type(number) is not int or number <= 0:
            raise ValueError(f"Некорректный порядковый номер задания: {number!r}")
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(f"Пустое условие задания {number}")
        item = {"task_num": number, "condition": condition}
        label = task.get("source_label")
        if not isinstance(label, str) or not label.strip() or label in labels:
            raise ValueError(f"Отсутствующая или повторная исходная метка: {label!r}")
        labels.add(label)
        item["source_label"] = label
        prepared.append(item)
    if sorted(item["task_num"] for item in prepared) != list(range(1, len(prepared) + 1)):
        raise ValueError("task_num должны идти от 1 до числа задач без повторов и пропусков")
    return sorted(prepared, key=lambda item: item["task_num"])


def classify_variant(context, tasks, client, *, exams_path, topics_path, cache_dir=None):
    """Классифицировать один вариант: один запрос либо ноль при точном кэше.

    tasks содержит task_num/condition/source_label, context задаёт вариант,
    исходный URL и пути. Возвращает {task_num: {exams_id, topics_id}}, где
    exams_id — числовой лист, topics_id — числовой лист или новая тема.
    Модель выбирает E/T-коды; ответ декодируется в раздельных пространствах
    справочников. Совместимые прежние числовые ответы тоже проверяются.

    classification.json в cache_dir или context.cache_dir связан с точной
    identity условий, справочников, варианта, провайдера и протокола. В нём
    хранятся исходный ответ и проверенные ID. Только проверенный результат
    атомарно заменяет кэш; ошибка не запускает дополнительный запрос.

    ValueError означает неверные данные, дерево или ответ модели. Ошибки
    API распространяются вызывающему коду, чтобы он отличал блокировку
    всего пакета от сбоя одного варианта. Семантика требует отдельного ревью.
    """
    exams_path, topics_path = Path(exams_path), Path(topics_path)
    exams = load_catalog(
        exams_path, name_column="name_of_Exam", parent_column="parent_of_exam", root_id=2,
    )
    topics = load_catalog(topics_path, name_column="name", parent_column="parent", root_id=1)
    prepared = prepare_variant_tasks(tasks)
    exam_leaves = allowed_categories(exams, "E")
    topic_leaves = allowed_categories(topics, "T")
    if not exam_leaves:
        raise ValueError("В ветке exams с корнем 2 нет допустимых конечных категорий")
    numeric_catalogues = {
        "exams": [{"id": row["id"], "path": row["path"]} for row in exam_leaves],
        "topics": [{"id": row["id"], "path": row["path"]} for row in topic_leaves],
    }
    exam_codes = {f"E{i:03d}": row["id"] for i, row in enumerate(exam_leaves, 1)}
    topic_codes = {f"T{i:03d}": row["id"] for i, row in enumerate(topic_leaves, 1)}
    payload = {
        "variant_key": context.variant_key,
        "tasks": prepared,
        "exams": [{"choice": code, "path": row["path"]} for code, row in zip(exam_codes, exam_leaves)],
        "topics": [{"choice": code, "path": row["path"]} for code, row in zip(topic_codes, topic_leaves)],
    }
    identity = {
        "version": VARIANT_CACHE_VERSION,
        "rules_version": VARIANT_RULES_VERSION,
        "prompt_version": VARIANT_PROMPT_VERSION,
        "prompt_sha256": _variant_fingerprint(VARIANT_PROMPT),
        "schema": VARIANT_SCHEMA,
        "variant_key": context.variant_key,
        "source_url": context.source_url,
        "provider_settings": client.settings.identity("classification"),
        "tasks_sha256": _variant_fingerprint(prepared),
        "catalogs_sha256": {
            "exams": hashlib.sha256(exams_path.read_bytes()).hexdigest(),
            "topics": hashlib.sha256(topics_path.read_bytes()).hexdigest(),
        },
        "allowed_categories_sha256": _variant_fingerprint(numeric_catalogues),
    }
    fingerprint = _variant_fingerprint(identity)
    numeric_identity = {
        **identity, "version": 1, "prompt_version": NUMERIC_VARIANT_PROMPT_VERSION,
        "prompt_sha256": _variant_fingerprint(NUMERIC_VARIANT_PROMPT), "schema": NUMERIC_VARIANT_SCHEMA,
    }
    cache_root = Path(cache_dir) if cache_dir is not None else Path(context.cache_dir)
    cache_path = cache_root / "classification.json"
    numbers = {item["task_num"] for item in prepared}

    def validate_response(response):
        """Декодировать E/T-метки и проверить полный набор задач варианта."""
        if not isinstance(response, dict) or set(response) != {"tasks"}:
            raise ValueError("Классификация должна содержать только JSON-поле tasks")
        if not isinstance(response["tasks"], list):
            raise ValueError("tasks должен быть списком")
        decoded = []
        for item in response["tasks"]:
            if not isinstance(item, dict) or set(item) != {"task_num", "exams_id", "topics_id"}:
                raise ValueError("Неверная структура классификации")
            exam, topic = item["exams_id"], item["topics_id"]
            if isinstance(exam, str):
                if exam not in exam_codes:
                    raise ValueError(f"Неизвестная метка exams: {exam}")
                exam = exam_codes[exam]
            if isinstance(topic, str) and topic in topic_codes:
                topic = topic_codes[topic]
            elif isinstance(topic, str) and re.fullmatch(r"[ET]\d+", topic):
                raise ValueError(f"Неизвестная или чужая метка topics: {topic}")
            decoded.append({"task_num": item["task_num"], "exams_id": exam, "topics_id": topic})
        return validate_classification(decoded, exams, topics, task_numbers=numbers)

    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("identity") in (identity, numeric_identity)
                and cached.get("fingerprint") == _variant_fingerprint(cached["identity"])
            ):
                result = validate_response(cached["response"])
                normalized = validate_classification(cached["tasks"], exams, topics, task_numbers=numbers)
                if result != normalized or cached.get("requests_used") != 1:
                    raise ValueError("Сохранённая классификация расходится с ответом модели")
                print(f"{context.variant_key}: классификация из кэша, новых запросов 0.", flush=True)
                return result
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
            print(f"{context.variant_key}: кэш классификации не прошёл проверку: {error}", flush=True)

    messages = [
        {"role": "system", "content": VARIANT_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    response = client.request_json(messages, stage="classification", debug_dir=context.debug_dir)
    try:
        result = validate_response(response)
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"{context.variant_key}: классификация не прошла проверку после одного запроса. "
            f"Прежний кэш сохранён; автоматического исправления нет. Причина: {error}"
        ) from error
    normalized = [{"task_num": number, **result[number]} for number in sorted(result)]
    atomic_json(cache_path, {
        "identity": identity,
        "fingerprint": fingerprint,
        "requests_used": 1,
        "response": response,
        "tasks": normalized,
    })
    return result


def classification_review_warnings(tasks, *, exams_path, topics_path):
    """Явные признаки спорной темы; это замечания, а не подмена выбора модели.

    Пилот выявил выбор тригонометрических тем для рациональных задач с
    параметром и текстовых арифметических тем для чтения диаграмм. Проверка
    не доказывает семантическую точность остальных классификаций.
    Возвращает список замечаний без изменения задач и без модельных вызовов;
    ошибки чтения и структуры справочников распространяются вызывающему коду.
    """
    exams = load_catalog(exams_path, name_column="name_of_Exam", parent_column="parent_of_exam", root_id=2)
    topics = load_catalog(topics_path, name_column="name", parent_column="parent", root_id=1)
    exam_paths = {r["id"]: r["path"] for r in allowed_categories(exams, "E")}
    topic_paths = {r["id"]: r["path"] for r in allowed_categories(topics, "T")}
    warnings = []
    trig = re.compile(
        r"\\(?:arc)?(?:sin|cos|tan|cot)\b|\b(?:sin|cos|tg|ctg|tan|cot|arctg|arcctg|arcsin|arccos)\b"
        r"|\b(?:tg|ctg)(?=[A-Z])|синус|косинус|тангенс", re.I,
    )
    for task in tasks:
        topic_path = topic_paths.get(task["topics_id"], "")
        exam_path = exam_paths.get(task["exams_id"], "")
        label = task["source_label"]
        if "тригонометр" in topic_path.casefold() and not trig.search(task["condition"]):
            warnings.append(f"{label}: выбрана тригонометрическая тема без тригонометрии в условии: {topic_path}")
        if "чтение графиков и диаграмм" in exam_path.casefold() and "текстовые задачи" in topic_path.casefold():
            warnings.append(f"{label}: чтение диаграммы отнесено к текстовым арифметическим задачам; проверьте topics_id")
    return warnings
