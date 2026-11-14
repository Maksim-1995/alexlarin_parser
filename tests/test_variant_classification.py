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
        variant_key="ege_2013_trvar1", source_url="https://alexlarin.net/ege/2013/trvar1.html",
        cache_dir=tmp_path / "cache", debug_dir=tmp_path / "debug",
    )


@pytest.fixture
def tasks():
    return [
        {"task_num": 1, "source_label": "B1", "condition": "Стоимость выросла на двадцать процентов."},
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

    def request_json(self, messages, *, stage, debug_dir):
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
    assert payload["variant_key"] == context.variant_key
    assert payload["tasks"] == tasks
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
    assert len(client.calls) == 2


@pytest.mark.parametrize("corruption", [
    "missing", "duplicate", "extra", "parent_exam", "foreign_exam", "unknown_exam",
    "parent_topic", "foreign_topic", "unknown_topic", "numeric_string", "list", "boolean", "null", "wrong_root",
])
def test_invalid_response_never_retries_or_replaces_previous_cache(context, tasks, catalogues, corruption):
    client = FakeClient()
    run(context, tasks, catalogues, client)
    cache = context.cache_dir / "classification.json"
    previous = cache.read_bytes()
    tasks[0]["condition"] += " Уточнение."
    records = client.response["tasks"]
    if corruption == "missing":
        records.pop()
    elif corruption == "duplicate":
        records[1]["task_num"] = 1
    elif corruption == "extra":
        records[0]["explanation"] = "Лишнее поле"
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
        client.response["variant_key"] = "ege_2013_trvar1"
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
