"""Изолированный совместимый классификатор Gemini для задач 1–19.

Основной пакетный поток использует service.classify_variant. Этот модуль
сохраняет исторические параметры, протокол и отпечаток кэша варианта 542;
он не является автоматическим запасным провайдером для DeepSeek.
"""

from functools import partial
import hashlib
import json
import os
from pathlib import Path

from alexlarin_parser.io import atomic_json
from alexlarin_parser.prompts import load_prompt
from .catalogues import (
    TASK_NUMBERS,
    build_category_choices,
    decode_category_choices,
    load_catalog,
    validate_classification,
)


MODEL = "gemini-3.5-flash-lite"
THINKING_LEVEL = "medium"
CACHE_VERSION = 3
RULES_VERSION = "math-exam-leaves-new-topics-v4"
REQUEST_TIMEOUT_MS = 120_000
MAX_OUTPUT_TOKENS = 16384
PROMPT = load_prompt("classification_gemini_legacy.txt")
SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "task_num": {"type": "integer"},
            "exam_choice": {"type": "string"},
            "topic_kind": {"type": "string", "enum": ["existing", "new"]},
            "topic_choice": {"type": "string"},
        },
        "required": ["task_num", "exam_choice", "topic_kind", "topic_choice"],
        "additionalProperties": False,
    },
}


def prepare_tasks(tasks):
    """Подготовить копии task_num/condition для старого набора строго 1–19.

    Сортирует результат по номеру, не меняя входные задачи. Неполный набор,
    повторные номера, неверные типы и пустые условия вызывают ValueError.
    Для произвольного варианта используется prepare_variant_tasks.
    """
    prepared = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Каждое задание должно быть словарём с task_num и condition")
        number = task.get("task_num")
        condition = task.get("condition")
        if type(number) is not int:
            raise ValueError("Номер задания должен быть целым числом")
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(f"Пустое условие задания {number}")
        prepared.append({"task_num": number, "condition": condition})
    numbers = [task["task_num"] for task in prepared]
    if len(numbers) != 19 or set(numbers) != TASK_NUMBERS:
        raise ValueError("Ожидаются ровно задания 1–19 без повторов")
    return sorted(prepared, key=lambda task: task["task_num"])


def request_gemini(prompt, *, model, schema, request_counter=None):
    """Совместимый Gemini-транспорт: один generate_content без повторов SDK.

    Возвращает необработанный текст ответа; JSON/ID проверяет classify_tasks.
    Ключ читается из GEMINI_API_KEY, счётчик растёт перед вызовом SDK.
    Основной пакетный поток использует общий прямой DeepSeek-клиент.
    """
    requests_used = 0
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Не задана переменная GEMINI_API_KEY")

        # Импорт здесь позволяет читать готовый кэш без SDK и API-ключа.
        from google import genai
        from google.genai import types

        with genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ) as client:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(
                    thinking_level=THINKING_LEVEL,
                ),
            )
            generate_content = client.models.generate_content
            requests_used += 1
            if request_counter is not None:
                request_counter.classification += 1
            response = generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        return response.text or ""
    finally:
        print(f"Запросов классификации в этом запуске: {requests_used}.", flush=True)


def classify_tasks(
    tasks, exams_path, topics_path, cache_path, *, model=MODEL,
    request=None, request_counter=None,
):
    """Совместимый классификатор набора 1–19 с историческим Gemini-кэшем.

    Возвращает {task_num: {exams_id, topics_id}}. Точный проверенный кэш
    исключает запрос; иначе выполняется один вызов request(prompt, model=)
    либо старого транспорта. Ошибка API поднимается как RuntimeError,
    неверный контракт — как ValueError; прежний кэш сохраняется при сбое.
    Для основного потока и произвольного числа задач есть classify_variant.
    """
    exams = load_catalog(
        exams_path,
        name_column="name_of_Exam",
        parent_column="parent_of_exam",
        root_id=2,
    )
    topics = load_catalog(
        topics_path,
        name_column="name",
        parent_column="parent",
        root_id=1,
    )
    exam_choices = build_category_choices(exams, "E")
    topic_choices = build_category_choices(topics, "T")

    payload = {
        "tasks": prepare_tasks(tasks),
        "exams": list(exam_choices),
        "topics": list(topic_choices),
    }
    identity = {
        "version": CACHE_VERSION,
        "rules_version": RULES_VERSION,
        "model": model,
        "thinking_level": THINKING_LEVEL,
        "prompt": PROMPT,
        "schema": SCHEMA,
        "request_settings": {
            "timeout": REQUEST_TIMEOUT_MS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "http_attempts": 1,
            "response_mime_type": "application/json",
        },
        # Подписи скрывают ID от модели. Для кэша сохраняем полную структуру
        # обоих CSV, включая ID и связи за пределами выбранной ветки.
        "catalogs": {
            "exams": [exams.records[number] for number in sorted(exams.records)],
            "topics": [topics.records[number] for number in sorted(topics.records)],
        },
        "input": payload,
    }
    # Исторический отпечаток включает пробелы JSON. Компактный fingerprint
    # нового протокола здесь изменил бы совместимость проверенных кэшей 542.
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    cache_path = Path(cache_path)

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(cached, dict):
                raise ValueError("Нужен JSON-объект")
            if (
                cached.get("version") == CACHE_VERSION
                and cached.get("rules_version") == RULES_VERSION
                and cached.get("fingerprint") == fingerprint
            ):
                result = validate_classification(cached["tasks"], exams, topics)
                choices = decode_category_choices(cached["choices"], exam_choices, topic_choices, topics)
                if validate_classification(choices, exams, topics) != result:
                    raise ValueError("Сохранённые ID и исходный выбор категорий расходятся")
                # В старом успешном кэше могло быть два обращения. Это история
                # прежнего запуска: чтение такого кэша сейчас не вызывает API.
                if type(cached.get("requests_used")) is not int or cached["requests_used"] not in (1, 2):
                    raise ValueError("Некорректный счётчик запросов классификации")
                print(f"Классификация прочитана из кэша: {cache_path}. Запросов классификации: 0.", flush=True)
                return result
            print("Кэш классификации устарел: нужна классификация по действующим правилам.", flush=True)
        except (ValueError, TypeError, KeyError) as error:
            print(f"Кэш классификации не прошёл проверку: {error}. Нужна новая классификация.", flush=True)

    custom_request = request is not None
    if request is None:
        options = {"schema": SCHEMA}
        if request_counter is not None:
            options["request_counter"] = request_counter
        request = partial(request_gemini, **options)

    prompt = PROMPT + "\nДанные:\n" + json.dumps(payload, ensure_ascii=False)
    print("Классифицируем все 19 заданий: запрос 1/1.", flush=True)
    try:
        response = request(prompt, model=model)
    except Exception as error:
        raise RuntimeError(
            "Не удалось получить классификацию от Gemini: ошибка API или сети. "
            "Проверьте GEMINI_API_KEY, квоту и подключение. "
            "Автоматического повтора нет; прежняя выгрузка сохранена. "
            f"Причина: {error}"
        ) from error
    finally:
        if custom_request:
            print("Вызовов переданной функции классификации: 1.", flush=True)
    try:
        selected = json.loads(response)
        items = decode_category_choices(selected, exam_choices, topic_choices, topics)
        result = validate_classification(items, exams, topics)
    except (ValueError, TypeError) as error:
        raise ValueError(
            "Классификация не прошла проверку после одного запроса. "
            "Автоматического исправления нет; прежние кэш и выгрузка сохранены. "
            f"Причина: {error}"
        ) from error

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {
        "version": CACHE_VERSION,
        "rules_version": RULES_VERSION,
        "fingerprint": fingerprint,
        "model": model,
        "thinking_level": THINKING_LEVEL,
        "requests_used": 1,
        "tasks": items,
        "choices": selected,
    }
    atomic_json(cache_path, cached)
    print(f"Классификация сохранена: {cache_path}", flush=True)
    return result
