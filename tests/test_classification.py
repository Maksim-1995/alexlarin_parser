"""Контракт классификатора проверяется без обращений к Gemini."""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import classification
from classification import (
    build_category_choices,
    classify_tasks,
    decode_category_choices,
    load_catalog,
    validate_classification,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXAM_LABEL = "E | Вектор > Векторы и операции с ними"
TOPIC_LABEL = (
    "T | Геометрия > Планиметрия > Векторы > "
    "Умножение вектора на число. Скалярное произведение векторов"
)
NEW_TOPIC = "Первообразная и определённый интеграл"


@pytest.fixture
def catalogs():
    exams = load_catalog(
        DATA_DIR / "exams.csv", name_column="name_of_Exam",
        parent_column="parent_of_exam", root_id=2,
    )
    topics = load_catalog(
        DATA_DIR / "topics.csv", name_column="name",
        parent_column="parent", root_id=1,
    )
    return exams, topics


@pytest.fixture
def tasks():
    return [
        {"task_num": n, "condition": "Найдите скалярное произведение векторов."}
        for n in range(1, 20)
    ]


@pytest.fixture
def response_items():
    return [
        {
            "task_num": n, "exam_choice": EXAM_LABEL,
            "topic_kind": "existing", "topic_choice": TOPIC_LABEL,
        }
        for n in range(1, 20)
    ]


@pytest.fixture
def numeric_items():
    return [
        {"task_num": n, "exams_id": 180, "topics_id": 526}
        for n in range(1, 20)
    ]


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "classification.json"


def run(tasks, cache, response_items, *, request=None, **kwargs):
    if request is None:
        request = lambda *args, **options: json.dumps(response_items)
    return classify_tasks(
        tasks, DATA_DIR / "exams.csv", DATA_DIR / "topics.csv", cache,
        request=request, **kwargs,
    )


def choices(catalogs):
    exams, topics = catalogs
    return build_category_choices(exams, "E"), build_category_choices(topics, "T")


def test_catalog_choices_use_all_exam_descendant_leaves_and_math_topic_leaves(catalogs):
    exam_choices, topic_choices = choices(catalogs)
    # Сохраняются пути родителей, но выбирать можно лишь конечные записи.
    assert exam_choices[EXAM_LABEL] == 180
    assert exam_choices["E | Планиметрия > Описанные окружности"] == 178
    assert exam_choices["E | Чтение графиков и диаграмм"] == 4
    assert 179 not in exam_choices.values()
    assert 8 not in exam_choices.values()
    assert 7 not in exam_choices.values()
    assert 34 not in exam_choices.values()  # Конечная категория ОГЭ.
    assert topic_choices[TOPIC_LABEL] == 526
    assert 98 not in topic_choices.values()
    assert 193 not in topic_choices.values()
    with (DATA_DIR / "exams.csv").open(encoding="utf-8-sig") as source:
        rows = {int(row["id"]): row for row in csv.DictReader(source)}
    parents = {int(row["parent_of_exam"]) for row in rows.values()}
    expected_exams = set()
    for number, row in rows.items():
        if number in parents:
            continue
        current = int(row["parent_of_exam"])
        while current and current != 2:
            current = int(rows[current]["parent_of_exam"])
        if current == 2:
            expected_exams.add(number)
    assert set(exam_choices.values()) == expected_exams
    with (DATA_DIR / "topics.csv").open(encoding="utf-8-sig") as source:
        rows = {int(row["id"]): row for row in csv.DictReader(source)}
    parents = {int(row["parent"]) for row in rows.values()}
    for number in topic_choices.values():
        assert number not in parents
        current = number
        while current and current != 1:
            current = int(rows[current]["parent"])
        assert current == 1


@pytest.mark.parametrize("exam_id", [4, 5, 178, 180])
def test_final_validation_accepts_direct_and_deep_math_leaves(catalogs, numeric_items, exam_id):
    numeric_items[0]["exams_id"] = exam_id
    result = validate_classification(numeric_items, *catalogs)
    assert result[1]["exams_id"] == exam_id


def test_children_are_checked_in_complete_catalog_even_with_reduced_candidates(catalogs, numeric_items):
    exams, topics = catalogs
    # Ребёнок 180 убран из кандидатов, но остаётся в исходном справочнике.
    # Из-за этого родитель 179 всё равно не превращается в лист.
    reduced = classification.Catalog(
        [row for row in exams if row["id"] != 180], exams.records, exams.root_id,
    )
    assert "E | Вектор" not in build_category_choices(reduced, "E")
    numeric_items[0]["exams_id"] = 179
    with pytest.raises(ValueError, match="недопустимый exams_id"):
        validate_classification(numeric_items, reduced, topics)


def test_exam_leaf_label_keeps_parent_path_when_candidates_are_already_reduced(catalogs):
    exams, _ = catalogs
    only_leaf = classification.Catalog(
        [row for row in exams if row["id"] == 180], exams.records, exams.root_id,
    )
    assert build_category_choices(only_leaf, "E") == {EXAM_LABEL: 180}


def test_other_branch_leaf_is_not_allowed_even_when_in_candidate_rows(catalogs, numeric_items):
    exams, topics = catalogs
    mixed = classification.Catalog(
        [*exams, exams.records[34]], exams.records, exams.root_id,
    )
    assert 34 not in build_category_choices(mixed, "E").values()
    numeric_items[0]["exams_id"] = 34
    with pytest.raises(ValueError, match="недопустимый exams_id"):
        validate_classification(numeric_items, mixed, topics)


def test_all_nineteen_are_sent_once_and_input_is_unchanged(tasks, cache, response_items):
    calls = []
    original = copy.deepcopy(tasks)

    def request(prompt, *, model):
        calls.append(prompt)
        payload = json.loads(prompt.split("Данные:\n", 1)[1])
        assert [task["task_num"] for task in payload["tasks"]] == list(range(1, 20))
        assert EXAM_LABEL in payload["exams"]
        assert TOPIC_LABEL in payload["topics"]
        assert "E | Вектор" not in payload["exams"]
        assert "E | Планиметрия" not in payload["exams"]
        assert "T | основы математического анализа" not in payload["topics"]
        return json.dumps(response_items)

    result = run(tasks, cache, response_items, request=request)
    assert len(calls) == 1
    assert result[2] == {"exams_id": 180, "topics_id": 526}
    assert len(result) == 19
    assert tasks == original


def test_new_topic_round_trips_response_and_cache(tasks, cache, response_items):
    response_items[7].update(topic_kind="new", topic_choice=NEW_TOPIC)
    result = run(tasks, cache, response_items)
    assert result[8] == {"exams_id": 180, "topics_id": NEW_TOPIC}
    assert type(result[2]["topics_id"]) is int
    cached = json.loads(cache.read_text(encoding="utf-8"))
    assert cached["tasks"][7]["topics_id"] == NEW_TOPIC
    assert cached["choices"] == response_items
    assert cached["choices"][7]["topic_kind"] == "new"
    assert cached["requests_used"] == 1
    assert not cache.with_suffix(".tmp").exists()

    def unexpected_request(*args, **kwargs):
        pytest.fail("Подходящий кэш должен исключать запросы к Gemini")

    assert run(tasks, cache, response_items, request=unexpected_request) == result


@pytest.mark.parametrize("bad_first", [
    "json", "unknown_exam", "parent_exam", "foreign_exam", "missing", "duplicate", "parent_topic",
])
def test_invalid_contract_is_repaired_for_all_tasks_in_second_request(
    tasks, cache, response_items, bad_first,
):
    wrong = copy.deepcopy(response_items)
    if bad_first == "unknown_exam":
        wrong[0]["exam_choice"] = TOPIC_LABEL
    elif bad_first == "parent_exam":
        wrong[0]["exam_choice"] = "E | Вектор"
    elif bad_first == "foreign_exam":
        wrong[0]["exam_choice"] = "E | Сараи; шины; печки"
    elif bad_first == "missing":
        wrong.pop()
    elif bad_first == "duplicate":
        wrong[-1]["task_num"] = 1
    elif bad_first == "parent_topic":
        wrong[0]["topic_choice"] = "T | Алгебра > основы математического анализа"
    first_response = "не JSON" if bad_first == "json" else json.dumps(wrong)
    calls = []

    def request(prompt, *, model):
        calls.append(prompt)
        assert all(f'"task_num": {n}, "condition"' in prompt for n in range(1, 20))
        return first_response if len(calls) == 1 else json.dumps(response_items)

    assert len(run(tasks, cache, response_items, request=request)) == 19
    assert len(calls) == 2
    assert "Предыдущий ответ" in calls[1]
    assert json.loads(cache.read_text(encoding="utf-8"))["requests_used"] == 2


def test_two_bad_responses_stop_without_saving_cache(tasks, cache, response_items):
    calls = []

    def request(prompt, *, model):
        calls.append(prompt)
        return "не JSON"

    with pytest.raises(ValueError, match="после двух запросов"):
        run(tasks, cache, response_items, request=request)
    assert len(calls) == 2
    assert not cache.exists()


def test_network_error_is_clear_and_does_not_retry(tasks, cache, response_items):
    calls = []

    def request(prompt, *, model):
        calls.append(prompt)
        raise TimeoutError("Тестовый тайм-аут")

    with pytest.raises(RuntimeError, match="Gemini|классификац|сети|API"):
        run(tasks, cache, response_items, request=request)
    assert len(calls) == 1
    assert not cache.exists()


@pytest.mark.parametrize("field,value", [
    ("exams_id", 179), ("exams_id", 8), ("exams_id", 7),
    ("exams_id", 34), ("exams_id", 2), ("exams_id", 999999),
    ("exams_id", 0), ("exams_id", -1), ("exams_id", "180"),
    ("exams_id", []), ("exams_id", {}), ("exams_id", None),
    ("exams_id", "Вектор"), ("exams_id", True), ("exams_id", 180.0),
    ("topics_id", 98), ("topics_id", 193), ("topics_id", 196),  # Лист физики.
    ("topics_id", 999999), ("topics_id", 0), ("topics_id", -1),
    ("topics_id", True), ("topics_id", 526.0), ("topics_id", [526]),
    ("topics_id", {}), ("topics_id", None), ("topics_id", ""),
    ("topics_id", "  "), ("topics_id", "526"), ("topics_id", "999999"),
    ("topics_id", "не найдено"), ("topics_id", "другое"),
    ("topics_id", "нет темы"), ("topics_id", "основы математического анализа"),
])
def test_final_validation_rejects_forbidden_ids_and_values(catalogs, numeric_items, field, value):
    numeric_items[0][field] = value
    with pytest.raises(ValueError):
        validate_classification(numeric_items, *catalogs)


@pytest.mark.parametrize("change", ["duplicate", "missing", "extra", "boolean_number", "not_dict", "not_list"])
def test_final_validation_requires_exactly_nineteen_records(catalogs, numeric_items, change):
    if change == "duplicate":
        numeric_items[-1]["task_num"] = 1
    elif change == "missing":
        numeric_items.pop()
    elif change == "extra":
        numeric_items[0]["explanation"] = "Лишнее поле"
    elif change == "boolean_number":
        numeric_items[0]["task_num"] = True
    elif change == "not_dict":
        numeric_items[0] = []
    elif change == "not_list":
        numeric_items = {"tasks": numeric_items}
    with pytest.raises(ValueError):
        validate_classification(numeric_items, *catalogs)


@pytest.mark.parametrize("name", [
    "", " ", "526", "-7", "1.5", "не найдено", "ДРУГОЕ", "нет темы",
    "основы математического анализа", "Математика", True, 999999, [], {}, None,
    "материальная точка. Система отсчета",  # Существующий лист физики.
    "Первообразная для задачи 8", "<b>Первообразная</b>",
    "Нет подходящей конечной темы", "Не найдена подходящая тема в справочнике",
    "Тема не найдена", "Не определено в каталоге",
    "Первообразная\x00", "Первообразная\x0b", "Первообразная\u200b",
    "Первообразная для задачи Ⅷ", "Вероятность ½",
])
def test_new_topic_rejects_invalid_names_and_parent_bypass(catalogs, response_items, name):
    response_items[0].update(topic_kind="new", topic_choice=name)
    with pytest.raises(ValueError):
        decode_category_choices(response_items, *choices(catalogs), topics=catalogs[1])


def test_new_name_matching_one_existing_leaf_becomes_real_id(catalogs, response_items):
    response_items[0].update(
        topic_kind="new",
        topic_choice=(
            "  УМНОЖЕНИЕ   ВЕКТОРА НА ЧИСЛО. "
            "СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ ВЕКТОРОВ  "
        ),
    )
    decoded = decode_category_choices(response_items, *choices(catalogs), topics=catalogs[1])
    assert decoded[0]["topics_id"] == 526


@pytest.mark.parametrize("number", [175, 93, 263])
def test_existing_leaf_name_is_resolved_before_restrictions_on_new_names(
    catalogs, response_items, number,
):
    # Существующие названия могут содержать цифры, формулы или быть длинными.
    # Ограничения новых тем не должны мешать восстановить настоящий ID.
    response_items[0].update(
        topic_kind="new", topic_choice=catalogs[1].records[number]["name"],
    )
    decoded = decode_category_choices(response_items, *choices(catalogs), topics=catalogs[1])
    assert decoded[0]["topics_id"] == number


@pytest.mark.parametrize("change", ["no_kind", "wrong_kind", "numeric_exam", "numeric_topic", "extra", "not_dict"])
def test_choice_contract_requires_exact_labels_and_explicit_topic_kind(catalogs, response_items, change):
    if change == "no_kind":
        response_items[0].pop("topic_kind")
    elif change == "wrong_kind":
        response_items[0]["topic_kind"] = "unknown"
    elif change == "numeric_exam":
        response_items[0]["exam_choice"] = 180
    elif change == "numeric_topic":
        response_items[0]["topic_choice"] = 526
    elif change == "extra":
        response_items[0]["topics_id"] = 526
    elif change == "not_dict":
        response_items[0] = False
    with pytest.raises(ValueError):
        decode_category_choices(response_items, *choices(catalogs), topics=catalogs[1])


def write_catalog(path, rows, *, exams=False):
    columns = ["id", "name_of_Exam", "parent_of_exam"] if exams else ["id", "name", "parent"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.mark.parametrize("invalid_rows", [
    [(1, "Математика", 0), (10, "Тема", 999)],
    [(1, "Математика", 0), (10, "Первый", 11), (11, "Второй", 10)],
    [(1, "Математика", 0), (10, "Лист", 1), (10, "Дубликат", 1)],
    [(1, "Математика", 10), (10, "Цикл через корень", 1)],
    [(1, "Математика", 999), (10, "Лист", 1)],
    [(1, "Математика", 0), (10, "Лист", 1), (20, "Другая ветка", 999)],
])
def test_catalog_rejects_invalid_structure_even_outside_selected_branch(tmp_path, invalid_rows):
    path = tmp_path / "topics.csv"
    write_catalog(path, invalid_rows)
    with pytest.raises(ValueError):
        load_catalog(path, name_column="name", parent_column="parent", root_id=1)


def test_new_name_cannot_randomly_choose_between_duplicate_leaf_names(tmp_path):
    path = tmp_path / "topics.csv"
    write_catalog(path, [
        (1, "Математика", 0), (10, "Алгебра", 1), (11, "Геометрия", 1),
        (20, "Одинаковая тема", 10), (21, "Одинаковая тема", 11),
    ])
    topics = load_catalog(path, name_column="name", parent_column="parent", root_id=1)
    selected = [{
        "task_num": 1, "exam_choice": EXAM_LABEL,
        "topic_kind": "new", "topic_choice": "Одинаковая тема",
    }]
    with pytest.raises(ValueError):
        decode_category_choices(selected, {EXAM_LABEL: 180}, build_category_choices(topics, "T"), topics=topics)


def test_one_math_leaf_is_unambiguous_even_if_physics_has_the_same_name(tmp_path):
    path = tmp_path / "topics.csv"
    write_catalog(path, [
        (1, "Математика", 0), (10, "Алгебра", 1), (20, "Векторы", 10),
        (100, "Физика", 0), (101, "Векторы", 100),
    ])
    topics = load_catalog(path, name_column="name", parent_column="parent", root_id=1)
    selected = [{
        "task_num": 1, "exam_choice": EXAM_LABEL,
        "topic_kind": "new", "topic_choice": "ВЕКТОРЫ",
    }]
    decoded = decode_category_choices(
        selected, {EXAM_LABEL: 180}, build_category_choices(topics, "T"), topics=topics,
    )
    assert decoded[0]["topics_id"] == 20


@pytest.mark.parametrize("setting", [
    "condition", "model", "thinking", "prompt", "schema", "version",
    "rules", "timeout", "output_tokens",
])
def test_significant_inputs_invalidate_cache(tasks, cache, response_items, monkeypatch, setting):
    run(tasks, cache, response_items)
    old_fingerprint = json.loads(cache.read_text(encoding="utf-8"))["fingerprint"]
    options = {}
    if setting == "condition":
        tasks[0]["condition"] += " Длины векторов известны."
    elif setting == "model":
        options["model"] = "test-other-model"
    elif setting == "thinking":
        monkeypatch.setattr(classification, "THINKING_LEVEL", "high")
    elif setting == "prompt":
        monkeypatch.setattr(classification, "PROMPT", classification.PROMPT + "\nУточнение.")
    elif setting == "schema":
        schema = copy.deepcopy(classification.SCHEMA)
        schema["description"] = "Изменённый контракт"
        monkeypatch.setattr(classification, "SCHEMA", schema)
    elif setting == "version":
        monkeypatch.setattr(classification, "CACHE_VERSION", classification.CACHE_VERSION + 1)
    elif setting == "rules":
        monkeypatch.setattr(classification, "RULES_VERSION", classification.RULES_VERSION + "-changed")
    elif setting == "timeout":
        monkeypatch.setattr(classification, "REQUEST_TIMEOUT_MS", classification.REQUEST_TIMEOUT_MS + 1)
    elif setting == "output_tokens":
        monkeypatch.setattr(classification, "MAX_OUTPUT_TOKENS", classification.MAX_OUTPUT_TOKENS + 1)
    calls = []

    def request(prompt, *, model):
        calls.append(prompt)
        return json.dumps(response_items)

    run(tasks, cache, response_items, request=request, **options)
    assert len(calls) == 1
    assert json.loads(cache.read_text(encoding="utf-8"))["fingerprint"] != old_fingerprint


@pytest.mark.parametrize("change", ["exam_id", "topic_id", "parent", "name"])
def test_catalog_identity_includes_ids_names_and_parent_links(tmp_path, tasks, change):
    exams_path = tmp_path / "exams.csv"
    topics_path = tmp_path / "topics.csv"
    cache_path = tmp_path / "classification.json"
    exams = [(1, "Экзамены", 0), (2, "математика профиль", 1), (179, "Вектор", 2)]
    topics = [(1, "Математика", 0), (10, "Алгебра", 1), (11, "Геометрия", 1), (526, "Векторы", 10)]
    write_catalog(exams_path, exams, exams=True)
    write_catalog(topics_path, topics)
    calls = []

    def request(prompt, *, model):
        calls.append(prompt)
        payload = json.loads(prompt.split("Данные:\n", 1)[1])
        topic = next(label for label in payload["topics"] if "Векторы" in label)
        return json.dumps([
            {"task_num": n, "exam_choice": payload["exams"][0], "topic_kind": "existing", "topic_choice": topic}
            for n in range(1, 20)
        ])

    classify_tasks(tasks, exams_path, topics_path, cache_path, request=request)
    old_fingerprint = json.loads(cache_path.read_text(encoding="utf-8"))["fingerprint"]
    if change == "exam_id":
        exams[-1] = (579, "Вектор", 2)
    elif change == "topic_id":
        topics[-1] = (626, "Векторы", 10)
    elif change == "parent":
        topics[-1] = (526, "Векторы", 11)
    elif change == "name":
        topics[-1] = (526, "Векторы и операции", 10)
    write_catalog(exams_path, exams, exams=True)
    write_catalog(topics_path, topics)
    result = classify_tasks(tasks, exams_path, topics_path, cache_path, request=request)
    assert len(calls) == 2
    assert json.loads(cache_path.read_text(encoding="utf-8"))["fingerprint"] != old_fingerprint
    if change == "exam_id":
        assert result[1]["exams_id"] == 579
    if change == "topic_id":
        assert result[1]["topics_id"] == 626


@pytest.mark.parametrize("corruption", [
    "old_version", "version_only", "previous_version", "old_rules", "forbidden_exam", "parent_topic", "unknown_topic", "invalid_new",
    "missing_choices", "choices_differ", "wrong_kind", "invalid_json",
])
def test_invalid_or_legacy_cache_is_never_accepted(tasks, cache, response_items, corruption):
    run(tasks, cache, response_items)
    cached = json.loads(cache.read_text(encoding="utf-8"))
    if corruption == "old_version":
        cached["version"] = 1
        cached["fingerprint"] = "legacy-classification-fingerprint"
        cached["tasks"][0].update(exams_id=179, topics_id=98)
    elif corruption == "version_only":
        cached["version"] = 1
    elif corruption == "previous_version":
        cached["version"] = classification.CACHE_VERSION - 1
    elif corruption == "old_rules":
        cached["rules_version"] = "direct-exams-math-leaves-new-topics-v3"
    elif corruption == "forbidden_exam":
        cached["tasks"][0]["exams_id"] = 179
    elif corruption == "parent_topic":
        cached["tasks"][0]["topics_id"] = 98
    elif corruption == "unknown_topic":
        cached["tasks"][0]["topics_id"] = 999999
    elif corruption == "invalid_new":
        cached["tasks"][0]["topics_id"] = "нет темы"
    elif corruption == "missing_choices":
        cached.pop("choices")
    elif corruption == "choices_differ":
        cached["choices"][0].update(topic_kind="new", topic_choice=NEW_TOPIC)
    elif corruption == "wrong_kind":
        cached["choices"][0]["topic_kind"] = "unknown"
    cache.write_text("не JSON" if corruption == "invalid_json" else json.dumps(cached), encoding="utf-8")
    calls = []

    def request(prompt, *, model):
        calls.append(prompt)
        return json.dumps(response_items)

    result = run(tasks, cache, response_items, request=request)
    assert len(calls) == 1
    assert result[1] == {"exams_id": 180, "topics_id": 526}


@pytest.mark.parametrize("keep_current_metadata", [False, True])
def test_old_direct_parent_cache_is_rejected_even_with_current_fingerprint(
    tasks, cache, response_items, keep_current_metadata,
):
    run(tasks, cache, response_items)
    cached = json.loads(cache.read_text(encoding="utf-8"))
    for task in cached["tasks"]:
        task["exams_id"] = 179
    for choice in cached["choices"]:
        choice["exam_choice"] = "E | Вектор"
    if not keep_current_metadata:
        cached["version"] = 3
        cached["rules_version"] = "direct-exams-math-leaves-new-topics-v3"
        cached["fingerprint"] = "old-direct-parent-fingerprint"
    cache.write_text(json.dumps(cached), encoding="utf-8")
    calls = []

    def request(prompt, *, model):
        calls.append(prompt)
        return json.dumps(response_items)

    result = run(tasks, cache, response_items, request=request)
    assert len(calls) == 1
    assert result[1] == {"exams_id": 180, "topics_id": 526}
    replaced = json.loads(cache.read_text(encoding="utf-8"))
    assert replaced["choices"] == response_items
    assert replaced["tasks"][0]["exams_id"] == 180


def test_failed_reclassification_preserves_previous_cache(tasks, cache, response_items):
    run(tasks, cache, response_items)
    original = cache.read_bytes()
    tasks[0]["condition"] += " Уточнение условия."
    with pytest.raises(ValueError, match="после двух запросов"):
        run(tasks, cache, response_items, request=lambda *args, **kwargs: "не JSON")
    assert cache.read_bytes() == original


def test_atomic_cache_replace_failure_keeps_previous_success(
    tasks, cache, response_items, monkeypatch,
):
    run(tasks, cache, response_items)
    original = cache.read_bytes()
    tasks[0]["condition"] += " Уточнение условия."
    original_replace = Path.replace
    replacements = []

    def fail_cache_replace(pending, destination):
        if Path(destination) == cache:
            replacements.append(pending)
            assert pending != cache
            assert cache.read_bytes() == original
            assert len(json.loads(pending.read_text(encoding="utf-8"))["tasks"]) == 19
            raise OSError("Тестовая ошибка записи на диск")
        return original_replace(pending, destination)

    monkeypatch.setattr(Path, "replace", fail_cache_replace)
    with pytest.raises(OSError, match="ошибка записи"):
        run(tasks, cache, response_items)
    assert len(replacements) == 1
    assert cache.read_bytes() == original
    assert not replacements[0].exists()


def test_cache_write_does_not_overwrite_another_pending_file(tasks, cache, response_items):
    # Старое общее имя .tmp могло принадлежать другому запуску.
    another_pending = cache.with_suffix(".tmp")
    another_pending.write_text("Чужая незавершённая запись", encoding="utf-8")

    assert len(run(tasks, cache, response_items)) == 19
    assert another_pending.read_text(encoding="utf-8") == "Чужая незавершённая запись"
    assert not list(cache.parent.glob(cache.name + ".*.tmp"))


@pytest.mark.parametrize("rows", [
    [(1, "Экзамены", 0), (2, "Профиль", 1), (10, "Вектор", 2), (11, "Потерянный родитель", 999)],
    [(1, "Экзамены", 0), (2, "Профиль", 10), (10, "Цикл", 2)],
    [(1, "Экзамены", 0), (2, "Профиль", 1), (10, "Вектор", 2), (10, "Дубликат", 2)],
])
def test_exam_catalog_structure_errors_are_also_rejected(tmp_path, rows):
    path = tmp_path / "exams.csv"
    write_catalog(path, rows, exams=True)
    with pytest.raises(ValueError):
        load_catalog(path, name_column="name_of_Exam", parent_column="parent_of_exam", root_id=2)


def test_all_subparts_are_kept_in_the_same_task_request(tasks, cache, response_items):
    tasks[0]["condition"] = "Общее условие. А) Докажите. Б) Вычислите. В) Сравните."
    calls = []

    def request(prompt, *, model):
        payload = json.loads(prompt.split("Данные:\n", 1)[1])
        calls.append(payload)
        assert payload["tasks"][0] == tasks[0]
        return json.dumps(response_items)

    assert len(run(tasks, cache, response_items, request=request)) == 19
    assert len(calls) == 1
    assert len(calls[0]["tasks"]) == 19


def test_missing_task_is_rejected_before_request(tasks, cache, response_items):
    tasks.pop()

    def unexpected(*args, **kwargs):
        pytest.fail("При неполном входе запрос недопустим")

    with pytest.raises(ValueError, match="ровно задания 1–19"):
        run(tasks, cache, response_items, request=unexpected)


@pytest.mark.parametrize("invalid_task", [None, [], 1, "условие"])
def test_malformed_input_task_is_rejected_clearly_before_request(
    tasks, cache, response_items, invalid_task,
):
    tasks[0] = invalid_task

    def unexpected(*args, **kwargs):
        pytest.fail("При некорректной структуре задания запрос недопустим")

    with pytest.raises(ValueError, match="задание должно быть словарём"):
        run(tasks, cache, response_items, request=unexpected)


def test_sdk_has_one_http_attempt_and_preserves_classifier_settings(monkeypatch, response_items):
    from google import genai

    options = {}
    calls = []

    class Client:
        def __init__(self, **kwargs):
            options.update(kwargs)
            self.models = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text=json.dumps(response_items))

    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr(genai, "Client", Client)
    classification.request_gemini("test", model=classification.MODEL, schema=classification.SCHEMA)
    assert len(calls) == 1
    assert options["http_options"].timeout == 120_000
    assert options["http_options"].retry_options.attempts == 1
    assert calls[0]["model"] == classification.MODEL
    assert calls[0]["config"].thinking_config.thinking_level.value.lower() == classification.THINKING_LEVEL
