"""Изоляция, деревья и кэш классификации без платных или сетевых вызовов."""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alexlarin_parser.classification import service as classification
from alexlarin_parser.classification.catalogues import load_catalog, normalize_catalog_id
from alexlarin_parser.classification.service import classify_variant
from alexlarin_parser.io import fingerprint


def write_catalog(path, rows, *, exams=False):
    columns = ["id", "name_of_Exam", "parent_of_exam"] if exams else ["id", "name", "parent"]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.fixture
def catalogues(tmp_path):
    exams_path, topics_path = tmp_path / "exams.csv", tmp_path / "topics.csv"
    write_catalog(exams_path, [
        (1, "ЕГЭ", ""), (2, "математика профиль", "1.0"),
        (10, "Уравнения", 2), (11, "Тригонометрические уравнения", 10),
        (12, "Практические задачи", 2), (3, "математика базовая", 1),
        (30, "Базовые уравнения", 3),
    ], exams=True)
    write_catalog(topics_path, [
        (1, "Математика", "NULL"), (10, "Алгебра", 1),
        (11, "Тригонометрические уравнения", "10.0"),
        (12, "Проценты", 10), (2, "Физика", 0), (20, "Скорость", 2),
    ])
    return exams_path, topics_path


@pytest.fixture
def context(tmp_path):
    return SimpleNamespace(
        base_dir=tmp_path,
        variant_key="ege_2013_trvar1", source_url="https://alexlarin.net/ege/2013/trvar1.html",
        cache_dir=tmp_path / "cache", debug_dir=tmp_path / "debug",
    )


@pytest.fixture
def tasks():
    return [
        {"task_num": 1, "source_label": "B1", "condition": "В магазине стоимость выросла на двадцать процентов."},
        {"task_num": 2, "source_label": "C1", "condition": "Решите $\\sin x = 0$. А) Найдите корни. Б) Отберите корни."},
    ]


class FakeClient:
    def __init__(self):
        self.model = "deepseek-flash"
        self.thinking = "disabled"
        self.calls = []
        self.failure = None
        self.response = {"tasks": [
            {"task_num": 1, "exams_id": 12, "topics_id": 12},
            {"task_num": 2, "exams_id": 11, "topics_id": 11},
        ]}
        self.settings = SimpleNamespace(identity=lambda stage: {
            "provider": "deepseek", "model": self.model, "thinking": self.thinking,
            "stage": stage, "max_tokens": 8192,
        })

    def request_json(self, messages, *, stage, debug_dir, validate=None):
        self.calls.append({"messages": messages, "stage": stage, "debug_dir": debug_dir})
        if self.failure:
            raise self.failure
        return copy.deepcopy(self.response)


def run(context, tasks, catalogues, client, **kwargs):
    return classify_variant(context, tasks, client, exams_path=catalogues[0], topics_path=catalogues[1], **kwargs)


def test_one_variant_one_request_then_cache_and_input_not_mutated(context, tasks, catalogues):
    original = copy.deepcopy(tasks)
    client = FakeClient()
    expected = {1: {"exams_id": 12, "topics_id": 12}, 2: {"exams_id": 11, "topics_id": 11}}
    assert run(context, tasks, catalogues, client) == expected
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["stage"] == "classification"
    payload = json.loads(call["messages"][1]["content"])
    assert "variant_key" not in payload
    assert payload["tasks"] == [{"task_num": task["task_num"], "condition": task["condition"]} for task in tasks]
    assert {row["choice"] for row in payload["exams"]} == {"E001", "E002"}
    assert {row["choice"] for row in payload["topics"]} == {"T001", "T002"}
    assert all("id" not in row for row in [*payload["exams"], *payload["topics"]])
    assert payload["exams"][0]["path"] == "Уравнения > Тригонометрические уравнения"
    assert tasks == original

    client.failure = AssertionError("Кэш обязан исключать новый запрос")
    assert run(context, tasks, catalogues, client) == expected
    assert len(client.calls) == 1
    cached = json.loads((context.cache_dir / "classification.json").read_text(encoding="utf-8"))
    assert cached["identity"]["provider_settings"]["provider"] == "deepseek"
    assert cached["identity"]["variant_key"] == context.variant_key


@pytest.mark.parametrize("location", ["root", "task", "both"])
@pytest.mark.parametrize("coded", [False, True])
def test_extra_keys_are_ignored_and_valid_response_is_cached(context, tasks, catalogues, location, coded):
    client = FakeClient()
    if location in {"root", "both"}:
        client.response.update(explanation={"confidence": 0.9}, variant_key=context.variant_key)
    if location in {"task", "both"}:
        client.response["tasks"][0].update(explanation="Комментарий модели", source_label="C99")
    if coded:
        for item in client.response["tasks"]:
            item.update(exams_id="E002" if item["task_num"] == 1 else "E001",
                        topics_id="T002" if item["task_num"] == 1 else "T001")
    original = copy.deepcopy(client.response)
    expected = {1: {"exams_id": 12, "topics_id": 12}, 2: {"exams_id": 11, "topics_id": 11}}
    assert run(context, tasks, catalogues, client) == expected
    cached = json.loads((context.cache_dir / "classification.json").read_text())
    assert cached["response"] == original  # Исходный ответ сохраняется для диагностики.
    assert cached["tasks"] == [{"task_num": number, **values} for number, values in expected.items()]
    client.failure = AssertionError("Готовый кэш должен читаться без модели")
    assert run(context, tasks, catalogues, client) == expected
    assert len(client.calls) == 1 and client.response == original


@pytest.mark.parametrize("field", ["tasks", "task_num", "exams_id", "topics_id"])
def test_extra_keys_cannot_replace_missing_classification_fields(context, tasks, catalogues, field):
    client = FakeClient()
    client.response["explanation"] = "Комментарий модели"
    client.response["tasks"][0]["explanation"] = "Комментарий модели"
    target = client.response if field == "tasks" else client.response["tasks"][0]
    del target[field]
    with pytest.raises(ValueError, match="поле tasks|структура классификации"):
        run(context, tasks, catalogues, client)
    assert len(client.calls) == 1
    assert not (context.cache_dir / "classification.json").exists()


def test_model_choices_are_decoded_in_separate_namespaces(context, tasks, catalogues):
    client = FakeClient()
    client.response = {"tasks": [
        {"task_num": 1, "exams_id": "E002", "topics_id": "T002"},
        {"task_num": 2, "exams_id": "E001", "topics_id": "T001"},
    ]}
    assert run(context, tasks, catalogues, client) == {
        1: {"exams_id": 12, "topics_id": 12}, 2: {"exams_id": 11, "topics_id": 11},
    }
    client.response["tasks"][0]["topics_id"] = "E002"
    tasks[0]["condition"] += " Изменённое условие."
    with pytest.raises(ValueError, match="чужая метка topics"):
        run(context, tasks, catalogues, client)
    assert len(client.calls) == 2


def test_content_cache_rebinds_renumbered_tasks_in_another_variant(context, tasks, catalogues):
    client = FakeClient()
    first = run(context, tasks, catalogues, client)
    current = SimpleNamespace(
        base_dir=context.base_dir, variant_key="ege_2014_trvar40",
        source_url="https://alexlarin.net/ege/2014/trvar40.html",
        cache_dir=context.base_dir / "other-cache", debug_dir=context.base_dir / "other-debug",
    )
    renumbered = [dict(tasks[1], task_num=1, source_label="B9"),
                  dict(tasks[0], task_num=2, source_label="C6")]
    client.failure = AssertionError("Тот же вход должен использовать кэш по содержимому")
    assert run(current, renumbered, catalogues, client) == {1: first[2], 2: first[1]}
    assert len(client.calls) == 1
    cached = json.loads((current.cache_dir / "classification.json").read_text())
    assert cached["identity"]["variant_key"] == current.variant_key
    assert cached["tasks"] == [{"task_num": 1, **first[2]}, {"task_num": 2, **first[1]}]
    # Отдельный fresh-кэш вынуждает запросить модель и сравнить реальные входы.
    client.failure = None
    assert run(current, renumbered, catalogues, client, cache_dir=current.base_dir / "fresh") == {1: first[2], 2: first[1]}
    assert client.calls[0]["messages"] == client.calls[1]["messages"]


@pytest.mark.parametrize("version", [1, 2])
def test_supported_old_variant_cache_keeps_original_task_numbers(context, tasks, catalogues, version):
    tasks = [dict(tasks[1], task_num=1), dict(tasks[0], task_num=2)]
    client = FakeClient()
    expected = run(context, tasks, catalogues, client)
    path = context.cache_dir / "classification.json"
    cached = json.loads(path.read_text())
    cached["identity"].update(version=version)
    if version == 1:
        cached["identity"].update(
            prompt_version=classification.NUMERIC_VARIANT_PROMPT_VERSION,
            prompt_sha256=fingerprint(classification.NUMERIC_VARIANT_PROMPT),
            schema=classification.NUMERIC_VARIANT_SCHEMA,
        )
    else:
        cached["identity"]["prompt_version"] = "deepseek-namespaced-leaves-v3"
    cached["fingerprint"] = fingerprint(cached["identity"])
    cached["response"] = {"tasks": [{"task_num": number, **values} for number, values in expected.items()]}
    path.write_text(json.dumps(cached))
    before = path.read_bytes()
    client.failure = AssertionError("Поддержанный старый кэш читается без нового API")
    assert run(context, tasks, catalogues, client, cache_dir=context.cache_dir) == expected
    assert len(client.calls) == 1 and path.read_bytes() == before


@pytest.mark.parametrize("corruption", ["integrity", "invalid_number", "invalid_id", "partial"])
def test_invalid_content_cache_is_not_used(context, tasks, catalogues, corruption):
    client = FakeClient()
    expected = run(context, tasks, catalogues, client)
    content_path = next((context.base_dir / "input" / "cache" / "model_responses" / "classification").glob("*.json"))
    cached = json.loads(content_path.read_text())
    if corruption == "invalid_number":
        cached["response"]["tasks"][0]["task_num"] = 999
    elif corruption == "invalid_id":
        cached["response"]["tasks"][0]["exams_id"] = 999
    elif corruption == "partial":
        cached["response"]["tasks"].pop()
    else:
        cached["response"]["tasks"][0]["topics_id"] = 11
    if corruption != "integrity":
        cached["response_sha256"] = fingerprint(cached["response"])
        cached["record_sha256"] = fingerprint({"response": cached["response"], "provenance": cached.get("provenance")})
    content_path.write_text(json.dumps(cached))
    current = copy.copy(context)
    current.cache_dir = context.base_dir / "other-cache"
    assert run(current, tasks, catalogues, client) == expected
    assert len(client.calls) == 2


def test_explicit_classification_cache_is_isolated_from_shared_and_repeatable(context, tasks, catalogues):
    client = FakeClient()
    expected = run(context, tasks, catalogues, client)
    fresh = context.base_dir / "fresh"
    assert run(context, tasks, catalogues, client, cache_dir=fresh) == expected
    assert len(client.calls) == 2
    client.failure = AssertionError("Повтор отдельного кэша не вызывает модель")
    assert run(context, tasks, catalogues, client, cache_dir=fresh) == expected


def test_non_model_image_metadata_does_not_invalidate_text_classification(context, tasks, catalogues):
    client = FakeClient()
    first = run(context, tasks, catalogues, client)
    tasks[0].update(image_name="another.png", answer="Другой опубликованный ответ")
    client.failure = AssertionError("Рисунки и ответы сейчас не передаются в классификацию")
    assert run(context, tasks, catalogues, client) == first


def test_changed_batch_input_does_not_reuse_one_matching_task(context, tasks, catalogues):
    client = FakeClient()
    run(context, tasks, catalogues, client)
    tasks[1]["condition"] += " Изменённый соседний вход пакета."
    run(context, tasks, catalogues, client)
    assert len(client.calls) == 2


def test_review_flags_trigonometric_topic_for_arithmetic_without_changing_ids(tasks, catalogues):
    from classification import classification_review_warnings
    given = [dict(task, exams_id=12, topics_id=11) for task in tasks]
    original = copy.deepcopy(given)
    warnings = classification_review_warnings(given, exams_path=catalogues[0], topics_path=catalogues[1])
    assert len(warnings) == 1 and "B1" in warnings[0]
    assert "C1" not in warnings[0]
    assert given == original


@pytest.mark.parametrize("size", [1, 19, 20, 21])
def test_task_count_is_variant_specific(context, catalogues, size):
    tasks = [{"task_num": n, "source_label": f"B{n}", "condition": "Найдите процент."} for n in range(1, size + 1)]
    client = FakeClient()
    client.response = {"tasks": [{"task_num": n, "exams_id": 12, "topics_id": 12} for n in range(1, size + 1)]}
    assert len(run(context, tasks, catalogues, client)) == size
    assert len(client.calls) == 1


@pytest.mark.parametrize("change", ["variant", "source_url", "condition", "label", "model", "thinking", "prompt", "schema", "rules", "catalogue"])
def test_significant_inputs_invalidate_only_classification_cache(context, tasks, catalogues, monkeypatch, change):
    client = FakeClient()
    run(context, tasks, catalogues, client)
    if change == "variant":
        context.variant_key = "ege_2013_trvar2"
    elif change == "source_url":
        context.source_url = "https://alexlarin.net/ege/2013/trvar2.html"
    elif change == "condition":
        tasks[0]["condition"] += " Первоначальная стоимость известна."
    elif change == "label":
        tasks[1]["source_label"] = "C2"
    elif change in {"model", "thinking"}:
        setattr(client, change, "changed")
    elif change == "prompt":
        monkeypatch.setattr(classification, "VARIANT_PROMPT", classification.VARIANT_PROMPT + "\nУточнение.")
    elif change == "schema":
        monkeypatch.setattr(classification, "VARIANT_SCHEMA", {**classification.VARIANT_SCHEMA, "description": "changed"})
    elif change == "rules":
        monkeypatch.setattr(classification, "VARIANT_RULES_VERSION", "changed")
    else:
        # Неизменившиеся листы: меняется полный справочник, включая чужую ветку.
        text = catalogues[1].read_text(encoding="utf-8").replace("Скорость", "Скорость движения")
        catalogues[1].write_text(text, encoding="utf-8")
    run(context, tasks, catalogues, client)
    assert len(client.calls) == (1 if change in {"variant", "source_url", "label"} else 2)


@pytest.mark.parametrize("corruption", [
    "missing", "duplicate", "wrong_number", "parent_exam", "foreign_exam", "unknown_exam",
    "parent_topic", "foreign_topic", "unknown_topic", "numeric_string", "list", "boolean", "null", "wrong_root",
])
def test_invalid_response_never_retries_or_replaces_previous_cache(context, tasks, catalogues, corruption):
    client = FakeClient()
    run(context, tasks, catalogues, client)
    cache = context.cache_dir / "classification.json"
    previous = cache.read_bytes()
    tasks[0]["condition"] += " Уточнение."
    client.response["explanation"] = "Дополнительное поле не отменяет проверки номеров и ID"
    for record in client.response["tasks"]:
        record["explanation"] = "Комментарий модели"
    records = client.response["tasks"]
    if corruption == "missing":
        records.pop()
    elif corruption == "duplicate":
        records[1]["task_num"] = 1
    elif corruption == "wrong_number":
        records[0]["task_num"] = 3
    elif corruption == "parent_exam":
        records[0]["exams_id"] = 10
    elif corruption == "foreign_exam":
        records[0]["exams_id"] = 30
    elif corruption == "unknown_exam":
        records[0]["exams_id"] = 999
    elif corruption == "parent_topic":
        records[0]["topics_id"] = 10
    elif corruption == "foreign_topic":
        records[0]["topics_id"] = 20
    elif corruption == "unknown_topic":
        records[0]["topics_id"] = 999
    elif corruption == "numeric_string":
        records[0]["topics_id"] = "12"
    elif corruption == "list":
        records[0]["exams_id"] = [12]
    elif corruption == "boolean":
        records[0]["task_num"] = True
    elif corruption == "null":
        records[0]["topics_id"] = None
    else:
        client.response = records
    with pytest.raises(ValueError, match="после одного запроса"):
        run(context, tasks, catalogues, client)
    assert len(client.calls) == 2
    assert cache.read_bytes() == previous


def test_new_meaningful_topic_permitted_and_existing_leaf_name_resolved(context, tasks, catalogues):
    client = FakeClient()
    client.response["tasks"][0]["topics_id"] = "Первообразная и определённый интеграл"
    client.response["tasks"][1]["topics_id"] = "  ТРИГОНОМЕТРИЧЕСКИЕ УРАВНЕНИЯ "
    result = run(context, tasks, catalogues, client)
    assert result[1]["topics_id"] == "Первообразная и определённый интеграл"
    assert result[2]["topics_id"] == 11
    prompt = client.calls[0]["messages"][0]["content"]
    assert "Если подходящая конечная тема есть" in prompt
    assert "Только если подходящего конечного листа нет" in prompt


@pytest.mark.parametrize("name", ["Алгебра", "Скорость", "другое", "нет темы", "Новая тема 123", "<b>Проценты</b>"])
def test_new_topic_cannot_bypass_existing_parents_or_name_contract(context, tasks, catalogues, name):
    client = FakeClient()
    client.response["tasks"][0]["topics_id"] = name
    with pytest.raises(ValueError):
        run(context, tasks, catalogues, client)
    assert len(client.calls) == 1
    assert not (context.cache_dir / "classification.json").exists()


def test_timeout_bubbles_up_for_run_policy_and_preserves_old_cache(context, tasks, catalogues):
    client = FakeClient()
    run(context, tasks, catalogues, client)
    cache = context.cache_dir / "classification.json"
    previous = cache.read_bytes()
    tasks[0]["condition"] += " Уточнение."
    client.failure = TimeoutError("Тайм-аут чтения DeepSeek")
    with pytest.raises(TimeoutError):
        run(context, tasks, catalogues, client)
    assert len(client.calls) == 2
    assert cache.read_bytes() == previous


def test_atomic_write_failure_preserves_old_cache(context, tasks, catalogues, monkeypatch):
    client = FakeClient()
    run(context, tasks, catalogues, client)
    cache = context.cache_dir / "classification.json"
    previous = cache.read_bytes()
    tasks[0]["condition"] += " Уточнение."
    original_replace = Path.replace

    def fail_replace(path, destination):
        if Path(destination) == cache:
            raise OSError("Ошибка записи")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="Ошибка записи"):
        run(context, tasks, catalogues, client)
    assert cache.read_bytes() == previous
    assert not list(context.cache_dir.glob("classification.json.*.tmp"))


def test_malformed_input_fails_before_request(context, tasks, catalogues):
    client = FakeClient()
    tasks[1]["source_label"] = "B1"
    with pytest.raises(ValueError, match="метка"):
        run(context, tasks, catalogues, client)
    assert client.calls == []


@pytest.mark.parametrize(("value", "expected"), [(" 2.0 ", 2), ("02", 2), ("20.000", 20), ("0", 0)])
def test_csv_ids_normalize_exact_integers(value, expected):
    assert normalize_catalog_id(value) == expected


@pytest.mark.parametrize("value", [True, 2.0, "2.1", "2e0", "-2", "2 0", "NaN", "", None])
def test_invalid_catalog_id_rejected(value):
    with pytest.raises(ValueError):
        normalize_catalog_id(value)


def test_normalized_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "topics.csv"
    write_catalog(path, [(1, "Математика", ""), ("2", "Арифметика", 1), ("2.0", "Дубликат", 1)])
    with pytest.raises(ValueError, match="повторный"):
        load_catalog(path, name_column="name", parent_column="parent", root_id=1)
