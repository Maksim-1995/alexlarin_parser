"""Пакетный поток с настоящими OCR-кэшами/Excel/ZIP и подменой платного API."""

import copy
import csv
import json
from pathlib import Path
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock
from zipfile import ZipFile

import pymupdf
import pytest
import requests
from openpyxl import load_workbook

from ai_requests import AIRequestCounter
import pipeline
from pipeline import plan_run, run_batch, selected_contexts
from provider import DeepSeekSettings, PaidCallsBlocked
from variant_sources import VariantContext
from alexlarin_parser.io import file_sha256
from alexlarin_parser import provider
from alexlarin_parser.paths import PROJECT_ROOT


def write_catalogues(base):
    data = base / "data"
    data.mkdir()
    for name, columns, rows in [
        ("exams.csv", ["id", "name_of_Exam", "parent_of_exam"], [
            (1, "ЕГЭ", 0), (2, "математика профиль", 1), (10, "Уравнения", 2),
            (11, "Тригонометрические уравнения", 10), (12, "Практические задачи", 2),
        ]),
        ("topics.csv", ["id", "name", "parent"], [
            (1, "Математика", 0), (10, "Алгебра", 1), (11, "Уравнения", 10),
            (12, "Проценты", 10),
        ]),
    ]:
        with (data / name).open("w", encoding="utf8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(columns)
            writer.writerows(rows)


def picture(colour):
    with pymupdf.open() as document:
        page = document.new_page(width=50, height=50)
        page.draw_rect(page.rect, color=colour, fill=colour)
        return page.get_pixmap().tobytes("png")


def make_source_pdf(path, number, *, note=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = picture((1, 0, 0) if number == 1 else (0, 0, 1))
    with pymupdf.open() as document:
        first = document.new_page(width=400, height=500)
        first.insert_text((25, 50), f"B1. Variant {number} task B1. {note}")
        first.insert_image(pymupdf.Rect(60, 90, 140, 170), stream=image)
        second = document.new_page(width=400, height=500)
        second.insert_text((25, 50), f"C1. Variant {number} task C1. {note}")
        document.save(path)


@pytest.fixture
def project(tmp_path, monkeypatch):
    write_catalogues(tmp_path)
    contexts = [VariantContext.from_url(f"https://alexlarin.net/ege/2013/trvar{n}.html", tmp_path)
                for n in (1, 2)]
    source_by_key = {}
    for context in contexts:
        make_source_pdf(context.pdf_path, context.number)
        answer_image = context.source_dir / "answers.png"
        answer_image.write_bytes(picture((0, 1, 0)))
        context.source_urls = {"html": context.source_url,
                               "pdf": context.source_url.replace(".html", ".pdf"),
                               "answer_images": [context.source_url.replace(".html", ".png")]}
        source_by_key[context.variant_key] = {
            "pdf_path": context.pdf_path, "page_count": 2,
            "source_labels": ["B1", "C1"], "page_labels": [["B1"], ["C1"]],
            "html_answers": {"B1": f"{context.number},5"}, "answer_images": [answer_image],
            "metadata": {"class": 11, "year": 2013, "date": None,
                         "title": f"Тренировочный вариант № {context.number}",
                         "code": context.code, "source_url": context.source_url},
            "warnings": [],
        }

    def prepare(context, *, local_only=False):
        return copy.deepcopy(source_by_key[context.variant_key])

    monkeypatch.setattr(pipeline, "prepare_sources", prepare)
    return tmp_path, contexts, source_by_key


class FakeClient:
    """Считает каждую фактическую попытку, включая ошибку и тайм-аут."""
    def __init__(self):
        self.settings = DeepSeekSettings(api_key="test-fixture-only", dpi=72)
        self.counter = AIRequestCounter()
        self.attempts = []
        self.calls = []
        self.failures = {}

    def request_json(self, messages, *, stage, debug_dir, validate=None):
        key = next(part for part in Path(debug_dir).parts if part.startswith("ege_"))
        number = int(re.fullmatch(r"ege_2013_trvar(\d+)", key)[1])
        page_number = None
        if stage == "ocr":
            page_number = int(re.search(r"'page_num': (\d+)", messages[0]["content"][0]["text"])[1])
        call = {"variant_key": key, "stage": stage, "page_num": page_number}
        self.calls.append(call)
        setattr(self.counter, stage, getattr(self.counter, stage) + 1)
        attempt = {"attempt": len(self.attempts) + 1, "stage": stage, "status": "RECEIVED",
                   "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                   "seconds": 0.01, "cost": None}
        self.attempts.append(attempt)
        failure = self.failures.get((key, stage, page_number))
        if failure:
            attempt.update(status="FAILED", usage=None, error=type(failure).__name__)
            raise failure
        if stage == "classification":
            payload = json.loads(messages[1]["content"])
            return {"tasks": [{"task_num": task["task_num"], "exams_id": 11, "topics_id": 11}
                              for task in payload["tasks"]]}
        label = "B1" if page_number == 1 else "C1"
        marker = "первого" if number == 1 else "второго"
        return {"page_num": page_number,
                "tasks": [{"source_label": label, "condition": f"Условие варианта {marker}: найдите $x^2$. "
                           "А) Вычислите. Б) Сравните.", "continuation": False}],
                "answers": {} if page_number == 1 else {"C1": rf"$\sqrt{{{number + 1}}}$"}, "issues": []}


def state_for(report, context):
    return next(state for state in report["variants"] if state["variant_key"] == context.variant_key)


def read_workbook(context):
    book = load_workbook(context.output_dir / "tasks.xlsx")
    try:
        rows = list(book["Tasks"].values)
        about = dict(zip(*list(book["about"].values)))
    finally:
        book.close()
    return rows, about


def test_real_batch_isolated_files_cold_p_plus_one_and_repeat_zero(project):
    base, contexts, _ = project
    client = FakeClient()
    report = run_batch(base, contexts, client=client)
    assert (report["ocr_calls"], report["classification_calls"]) == (4, 2)
    assert report["status"] == "NEEDS_REVIEW"
    assert report["usage"] == {"prompt_tokens": 60, "completion_tokens": 30, "total_tokens": 90,
                               "unknown_usage_attempts": 0, "cost": None}
    image_names = []
    for context in contexts:
        state = state_for(report, context)
        assert state["status"] == "NEEDS_REVIEW"
        assert state["result_current_attempt"] is True
        assert (state["page_count"], state["task_count"], state["image_count"], state["answers_count"]) == (2, 2, 1, 2)
        assert (state["ocr_calls"], state["classification_calls"]) == (2, 1)
        rows, about = read_workbook(context)
        assert rows[0] == ("task_num", "condition", "image_name", "solution", "answer", "exams_id", "topics_id")
        assert [row[0] for row in rows[1:]] == [1, 2]
        marker = "первого" if context.number == 1 else "второго"
        assert all(f"варианта {marker}" in row[1] for row in rows[1:])
        assert rows[1][4] == f"{context.number},5"
        assert rows[2][4] == rf"$\sqrt{{{context.number + 1}}}$"
        assert "<br>" in rows[1][1]
        assert about["year"] == 2013
        assert about["source_url"] == context.source_url
        assert about["date"] is None
        assert len(state["result"]["image_files"]) == 1
        image = Path(state["result"]["image_files"][0])
        assert image.parent == context.output_dir
        image_names.append(image.name)
        with ZipFile(context.archive_path) as archive:
            assert set(archive.namelist()) == {"tasks.xlsx", image.name}
            assert archive.testzip() is None
            assert archive.read(image.name) == image.read_bytes()
    # Одинаковые внутренние номера рисунков допустимы лишь в разных каталогах.
    assert image_names[0] == image_names[1]
    assert (contexts[0].output_dir / image_names[0]).read_bytes() != (contexts[1].output_dir / image_names[1]).read_bytes()
    manifest = json.loads(Path(report["manifest_path"]).read_text(encoding="utf8"))
    assert manifest["variants"][0]["result"]["image_files"] == state_for(report, contexts[0])["result"]["image_files"]
    assert all(Path(state["debug_dir"], "verification.json").is_file() for state in report["variants"])

    warmed = FakeClient()
    repeated = run_batch(base, contexts, client=warmed)
    assert (repeated["ocr_calls"], repeated["classification_calls"]) == (0, 0)
    assert warmed.calls == []
    assert all(state["previous_result"]["excel_exists"] for state in repeated["variants"])


@pytest.mark.parametrize("change", ["catalogue", "one_pdf"])
def test_catalogue_only_reclassifies_and_pdf_change_stays_in_one_variant(project, change):
    base, contexts, source_by_key = project
    run_batch(base, contexts, client=FakeClient())
    if change == "catalogue":
        topics = base / "data" / "topics.csv"
        topics.write_text(topics.read_text(encoding="utf8").replace("Проценты", "Проценты и пропорции"), encoding="utf8")
    else:
        changed = contexts[0].source_dir / "changed.pdf"
        make_source_pdf(changed, 1, note="changed source")
        source_by_key[contexts[0].variant_key]["pdf_path"] = changed
    client = FakeClient()
    report = run_batch(base, contexts, client=client)
    if change == "catalogue":
        assert (report["ocr_calls"], report["classification_calls"]) == (0, 2)
    else:
        assert (state_for(report, contexts[0])["ocr_calls"], state_for(report, contexts[1])["ocr_calls"]) == (2, 0)
        # Текст задач не изменился: классификация остаётся совместимой.
        assert report["classification_calls"] == 0


def test_timeout_preserves_previous_result_and_page_then_resume_missing_only(project):
    base, contexts, source_by_key = project
    run_batch(base, contexts, client=FakeClient())
    first = contexts[0]
    previous_excel = (first.output_dir / "tasks.xlsx").read_bytes()
    previous_archive = first.archive_path.read_bytes()
    changed = first.source_dir / "changed.pdf"
    make_source_pdf(changed, 1, note="changed source")
    source_by_key[first.variant_key]["pdf_path"] = changed
    client = FakeClient()
    client.failures[(first.variant_key, "ocr", 2)] = TimeoutError("Тестовый тайм-аут")
    failed = run_batch(base, contexts, client=client)
    state = state_for(failed, first)
    assert state["status"] == "FAILED"
    assert state["stage"] == "OCR"
    assert state["result"] is None and state["result_current_attempt"] is False
    assert "TimeoutError" in state["error"]
    assert state["previous_result"]["excel_exists"] is True
    assert (first.output_dir / "tasks.xlsx").read_bytes() == previous_excel
    assert first.archive_path.read_bytes() == previous_archive
    assert (first.cache_dir / "ocr" / "page_001.json").is_file()
    assert state["usage"]["unknown_usage_attempts"] == 1
    assert state_for(failed, contexts[1])["status"] == "NEEDS_REVIEW"

    resumed_contexts = selected_contexts(base, resume=failed["manifest_path"], local_only=True)
    resumed_client = FakeClient()
    resumed = run_batch(base, resumed_contexts, client=resumed_client, resume_from=failed["manifest_path"])
    assert (resumed["ocr_calls"], resumed["classification_calls"]) == (1, 0)
    assert resumed_client.calls == [{"variant_key": first.variant_key, "stage": "ocr", "page_num": 2}]
    assert state_for(resumed, first)["result_current_attempt"] is True


def test_fresh_failed_attempt_resumes_its_isolated_pages(project):
    base, contexts, _ = project
    first = contexts[0]
    client = FakeClient()
    client.failures[(first.variant_key, "ocr", 2)] = TimeoutError("Тестовый тайм-аут")
    failed = run_batch(base, [first], client=client, fresh_ai=True)
    cache_dir = Path(failed["variants"][0]["cache_dir"])
    assert (cache_dir / "ocr" / "page_001.json").is_file()
    assert not (first.cache_dir / "ocr" / "page_001.json").exists()
    resumed_contexts = selected_contexts(base, resume=failed["manifest_path"], local_only=True)
    resumed_client = FakeClient()
    resumed = run_batch(base, resumed_contexts, client=resumed_client, resume_from=failed["manifest_path"])
    assert (resumed["ocr_calls"], resumed["classification_calls"]) == (1, 1)
    assert resumed["variants"][0]["cache_dir"] == str(cache_dir)


def test_authorization_failure_stops_new_model_attempts_for_batch(project):
    base, contexts, _ = project
    client = FakeClient()
    client.failures[(contexts[0].variant_key, "ocr", 1)] = PaidCallsBlocked("DeepSeek HTTP 402: недостаточно средств")
    report = run_batch(base, contexts, client=client)
    assert report["status"] == "FAILED"
    assert (report["ocr_calls"], report["classification_calls"]) == (1, 0)
    assert len(client.calls) == 1
    assert [state["status"] for state in report["variants"]] == ["FAILED", "FAILED"]
    assert all("402" in state["error"] for state in report["variants"])
    assert state_for(report, contexts[1])["attempts"] == []
    assert report["usage"]["unknown_usage_attempts"] == 1
    assert report["usage"]["cost"] is None


def http_response(data, status=200):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(data).encode("utf-8")
    return response


def completion(data):
    return http_response({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(data)}}]})


@pytest.mark.parametrize("paid_status", [None, 401, 403])
def test_paid_block_does_not_stop_recovered_free_provider_in_next_variant(project, monkeypatch, paid_status):
    base, contexts, source_by_key = project
    now = [100.0]
    monkeypatch.setattr(provider.time, "monotonic", lambda: now[0])

    def prepare(context, *, local_only=False):
        if context == contexts[1]:
            now[0] += 61  # Бесплатный провайдер восстановился к следующему варианту.
        return copy.deepcopy(source_by_key[context.variant_key])

    monkeypatch.setattr(pipeline, "prepare_sources", prepare)
    settings = DeepSeekSettings(
        api_key="test-paid-key" if paid_status else "", dpi=72,
        free_providers=(provider._FreeProvider("gemini", "https://free.example/v1", "test-free-key", "vision", "text"),),
    )
    response_client = FakeClient()
    responses = [http_response({"error": "busy"}, 503)]
    if paid_status:
        responses.append(http_response({"error": "denied"}, paid_status))
    for stage, page in (("ocr", 1), ("ocr", 2), ("classification", None)):
        messages = ([{"content": [{"text": f"'page_num': {page}"}]}] if stage == "ocr" else
                    [{}, {"content": json.dumps({"variant_key": contexts[1].variant_key,
                                                 "tasks": [{"task_num": 1}, {"task_num": 2}]})}])
        responses.append(completion(response_client.request_json(messages, stage=stage, debug_dir=contexts[1].debug_dir)))
    session = Mock()
    session.post.side_effect = responses
    client = provider.DeepSeekClient(settings, session=session,
                                     rate_limiter=provider.RequestRateLimiter(clock=lambda: now[0]))
    report = run_batch(base, contexts, client=client)
    assert report["variants"][0]["status"] == "FAILED"
    assert report["variants"][1]["status"] == "NEEDS_REVIEW"
    assert report["variants"][1]["result_current_attempt"] is True
    assert [a["provider"] for a in report["variants"][1]["attempts"]] == ["gemini"] * 3
    assert session.post.call_count == client.counter.total == 4 + bool(paid_status)
    assert report["ocr_calls"] == 3 + bool(paid_status) and report["classification_calls"] == 1
    assert sum(a["provider"] == "deepseek-direct" for a in client.attempts) == bool(paid_status)


@pytest.mark.parametrize("extra_keys", [False, True])
def test_local_variant40_real_stages_use_free_chain_without_paid_key(tmp_path, monkeypatch, extra_keys):
    """Реальные источники/ответы №40 читаются, все новые файлы — в tmp_path."""
    original = VariantContext.from_url("https://alexlarin.net/ege/2014/trvar40.html", PROJECT_ROOT)
    if not original.source_dir.is_dir() or not (original.cache_dir / "classification.json").is_file():
        pytest.skip("Локальные источники и сохранённые ответы №40 отсутствуют")
    context = VariantContext.from_url(original.source_url, tmp_path)
    shutil.copytree(original.source_dir, context.source_dir)
    shutil.copytree(PROJECT_ROOT / "data", tmp_path / "data")
    page_responses = [json.loads(path.read_text())["response"]
                      for path in sorted((original.cache_dir / "ocr").glob("page_*.json"))]
    original_classification = json.loads((original.cache_dir / "classification.json").read_text())
    classification = original_classification["response"]
    expected_categories = {task["task_num"]: (task["exams_id"], task["topics_id"])
                           for task in original_classification["tasks"]}
    # Сохранённый ответ использовал номера исходного варианта. Подставной
    # транспорт нового запроса возвращает номера стабильного порядка условий.
    original_tasks = json.loads((original.cache_dir / "conditions.json").read_text())["tasks"]
    request_numbers = {task["task_num"]: number for number, task in
                       enumerate(sorted(original_tasks, key=lambda task: task["condition"]), 1)}
    for task in classification["tasks"]:
        task["task_num"] = request_numbers[task["task_num"]]
    assert len(page_responses) == 3
    if extra_keys:
        for data in [*page_responses, classification]:
            data["explanation"] = "Дополнительное поле корня"
            for task in data["tasks"]:
                task["explanation"] = "Дополнительное поле задачи"
    # Создание настоящего клиента остаётся в run_batch → from_env.
    monkeypatch.setattr(provider.os, "environ", {
        "DEEPSEEK_API_KEY": "", "OCR_DPI": "72",
        "GEMINI_API_KEY": "test-free-key", "GEMINI_OCR_MODEL": "test-vision",
        "NVIDIA_API_KEY": "test-nvidia-key", "NVIDIA_CLASSIFICATION_MODEL": "test-text",
        "GEMINI_RPM": "3", "GEMINI_RPD": "3", "NVIDIA_RPM": "1", "NVIDIA_RPD": "1",
    })
    session = Mock()
    session.post.side_effect = [completion(data) for data in [*page_responses, classification]]
    monkeypatch.setattr(provider.requests, "Session", lambda: session)
    report = run_batch(tmp_path, [context], local_only=True)
    state = report["variants"][0]
    assert state["error"] is None and state["result_current_attempt"] is True
    assert report["provider"] == "fallback-chain"
    assert (state["task_count"], state["page_count"], state["download_count"]) == (20, 3, 0)
    assert (report["ocr_calls"], report["classification_calls"]) == (3, 1)
    assert [a["provider"] for a in state["attempts"]] == ["gemini"] * 3 + ["nvidia-nim"]
    assert session.post.call_count == 4
    quotas = json.loads((tmp_path / "input" / "cache" / "provider_quotas.json").read_text())
    assert sorted(len(record["requests"]) for record in quotas["providers"].values()) == [1, 3]
    assert all("thinking" not in call.kwargs["json"] for call in session.post.call_args_list)
    ocr_content = session.post.call_args_list[0].kwargs["json"]["messages"][0]["content"]
    assert any(part["type"] == "image_url" for part in ocr_content)
    payload = json.loads(session.post.call_args_list[-1].kwargs["json"]["messages"][1]["content"])
    assert "variant_key" not in payload and len(payload["tasks"]) == 20
    assert all("source_label" not in task for task in payload["tasks"])
    rows, about = read_workbook(context)
    assert len(rows) == 21 and about["year"] == 2014 and about["source_url"] == original.source_url
    assert [row[5:7] for row in rows[1:]] == [expected_categories[number] for number in range(1, 21)]
    session.close.assert_called_once()


def test_local_variant40_old_caches_remain_usable_without_http(tmp_path):
    original = VariantContext.from_url("https://alexlarin.net/ege/2014/trvar40.html", PROJECT_ROOT)
    if not original.source_dir.is_dir() or not (original.cache_dir / "classification.json").is_file():
        pytest.skip("Локальные источники и сохранённые ответы №40 отсутствуют")
    context = VariantContext.from_url(original.source_url, tmp_path)
    shutil.copytree(original.source_dir, context.source_dir)
    shutil.copytree(original.cache_dir, context.cache_dir)
    shutil.copytree(PROJECT_ROOT / "data", tmp_path / "data")
    checked = [context.cache_dir / "classification.json", *(context.cache_dir / "ocr").glob("page_*.json")]
    before = {path: path.read_bytes() for path in checked}
    session = Mock()
    session.post.side_effect = AssertionError("Совместимые старые кэши не требуют HTTP")
    quota_path = tmp_path / "input" / "cache" / "provider_quotas.json"
    client = provider.DeepSeekClient(DeepSeekSettings(
        api_key="", rate_limits=(provider.ProviderLimits("deepseek-direct", rpm=1, rpd=1),),
        rate_limit_path=quota_path,
    ), session=session)
    report = run_batch(tmp_path, [context], client=client, local_only=True)
    assert report["variants"][0]["error"] is None
    assert report["variants"][0]["task_count"] == 20
    assert (report["ocr_calls"], report["classification_calls"]) == (0, 0)
    session.post.assert_not_called()
    assert not quota_path.exists()
    assert all(path.read_bytes() == data for path, data in before.items())


def test_classification_results_keep_variant_and_task_binding_when_completed_in_reverse(project):
    base, contexts, _ = project
    first_waiting, second_done = Event(), Event()
    completed = []

    class OutOfOrderSession:
        def __init__(self, index):
            self.index = index

        def post(self, url, **kwargs):
            if self.index == 0:
                first_waiting.set()
                assert second_done.wait(timeout=3)
            return completion({"tasks": [{"task_num": number, "exams_id": 10 + number, "topics_id": 10 + number}
                                          for number in (2, 1)]})

    settings = DeepSeekSettings(
        api_key="test-shared-key", rate_limits=(provider.ProviderLimits("deepseek-direct", rpm=2, rpd=2),),
        rate_limit_path=base / "input" / "cache" / "provider_quotas.json",
    )
    clients = [provider.DeepSeekClient(settings, session=OutOfOrderSession(index)) for index in range(2)]
    tasks = [
        [{"task_num": 1, "source_label": "B1", "condition": "Явное условие первого варианта"},
         {"task_num": 2, "source_label": "C1", "condition": "Алгебра первого варианта"}],
        [{"task_num": 1, "source_label": "B9", "condition": "Алгебра второго варианта"},
         {"task_num": 2, "source_label": "C6", "condition": "Явное условие второго варианта"}],
    ]

    def classify(index):
        if index == 1:
            assert first_waiting.wait(timeout=3)
        result = pipeline.classify_variant(
            contexts[index], tasks[index], clients[index],
            exams_path=base / "data" / "exams.csv", topics_path=base / "data" / "topics.csv",
        )
        completed.append(index)
        if index == 1:
            second_done.set()
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(classify, index) for index in range(2)]
        results = [future.result(timeout=5) for future in futures]
    assert completed == [1, 0]
    assert results == [
        {1: {"exams_id": 12, "topics_id": 12}, 2: {"exams_id": 11, "topics_id": 11}},
        {1: {"exams_id": 11, "topics_id": 11}, 2: {"exams_id": 12, "topics_id": 12}},
    ]
    assert sum(client.counter.total for client in clients) == 2
    for context, result in zip(contexts, results):
        cached = json.loads((context.cache_dir / "classification.json").read_text())
        assert cached["identity"]["variant_key"] == context.variant_key
        assert cached["tasks"] == [{"task_num": number, **result[number]} for number in (1, 2)]


def test_invalid_catalogue_blocks_before_sources_or_paid_api(project, monkeypatch):
    base, contexts, _ = project
    (base / "data" / "topics.csv").write_text("id,name,parent\n1,Математика,999\n2,Проценты,1\n", encoding="utf8")

    def unexpected(*args, **kwargs):
        pytest.fail("Невалидный справочник должен остановить запуск до загрузки исходников")

    monkeypatch.setattr(pipeline, "prepare_sources", unexpected)
    client = FakeClient()
    report = run_batch(base, contexts, client=client)
    assert report["status"] == "FAILED"
    assert client.calls == []
    assert all("Справочники" in state["error"] for state in report["variants"])


def test_one_source_failure_does_not_block_independent_variant(project, monkeypatch):
    base, contexts, source_by_key = project

    def prepare(context, *, local_only=False):
        if context.number == 1:
            raise FileNotFoundError("Недоступен PDF варианта 1")
        return copy.deepcopy(source_by_key[context.variant_key])

    monkeypatch.setattr(pipeline, "prepare_sources", prepare)
    client = FakeClient()
    report = run_batch(base, contexts, client=client)
    assert state_for(report, contexts[0])["status"] == "FAILED"
    assert state_for(report, contexts[0])["stage"] == "SOURCES"
    assert state_for(report, contexts[1])["status"] == "NEEDS_REVIEW"
    assert (report["ocr_calls"], report["classification_calls"]) == (2, 1)
    assert {call["variant_key"] for call in client.calls} == {contexts[1].variant_key}


def test_plan_and_resume_selection_need_no_client_or_archive_network(project, monkeypatch, tmp_path):
    base, contexts, _ = project

    def forbidden(*args, **kwargs):
        pytest.fail("План и выбор --resume не должны обращаться к модели или архиву")

    monkeypatch.setattr(pipeline, "DeepSeekClient", forbidden)
    monkeypatch.setattr(pipeline, "load_catalogue", forbidden)
    plan = plan_run(base, contexts)
    assert plan["paid_calls"] == 0
    assert [row["variant_key"] for row in plan["variants"]] == [c.variant_key for c in contexts]
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"variants": list(reversed(plan["variants"]))}), encoding="utf8")
    resumed = selected_contexts(base, resume=path)
    assert [c.variant_key for c in resumed] == [c.variant_key for c in reversed(contexts)]


def write_category_correction(base, context, *, document=None):
    """Ручная правка относится к источнику и исходному ответу FakeClient."""
    path = base / "harness" / "corrections" / f"{context.variant_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if document is None:
        document = {
            "version": 1, "variant_key": context.variant_key,
            "source_url": context.source_url, "pdf_sha256": file_sha256(context.pdf_path),
            "corrections": [{
                "source_label": "B1", "field": "topics_id", "old": 11, "new": 12,
                "reason": "Проверенная категория текущего условия",
                "evidence": ["source PDF, page 1; full topics tree"],
                "method": "manual-source-and-catalogue-review",
            }],
        }
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path, document


def forbid_model(client):
    def unexpected(*args, **kwargs):
        pytest.fail("Ручные категории должны применяться поверх кэша без новых API-вызовов")
    client.request_json = unexpected
    return client


def test_manual_category_correction_uses_raw_cache_and_keeps_other_variant(project):
    base, contexts, _ = project
    run_batch(base, contexts, client=FakeClient())
    first, second = contexts
    original_cache = {p.name: p.read_bytes() for p in first.cache_dir.rglob("*.json")}
    other_excel = (second.output_dir / "tasks.xlsx").read_bytes()
    other_zip = second.archive_path.read_bytes()
    path, document = write_category_correction(base, first)

    report = run_batch(base, [first], client=forbid_model(FakeClient()))
    state = report["variants"][0]
    assert report["ocr_calls"] == report["classification_calls"] == 0
    rows, _ = read_workbook(first)
    assert rows[1][6] == 12 and rows[2][6] == 11
    assert state["corrections"] == {
        "path": str(path), "sha256": file_sha256(path), "records": document["corrections"],
    }
    verification = json.loads((Path(state["debug_dir"]) / "verification.json").read_text())
    assert verification["corrections"] == state["corrections"]
    assert {p.name: p.read_bytes() for p in first.cache_dir.rglob("*.json")} == original_cache
    assert (second.output_dir / "tasks.xlsx").read_bytes() == other_excel
    assert second.archive_path.read_bytes() == other_zip


@pytest.mark.parametrize("problem", [
    "variant", "url", "pdf", "old", "label", "field", "duplicate", "evidence", "method",
    "parent_topic", "unknown_exam", "placeholder", "existing_topic_name", "invalid_json",
])
def test_invalid_manual_correction_preserves_previous_output_and_cache(project, problem):
    base, contexts, _ = project
    first = contexts[0]
    run_batch(base, [first], client=FakeClient())
    previous_excel = (first.output_dir / "tasks.xlsx").read_bytes()
    previous_zip = first.archive_path.read_bytes()
    original_cache = {p.name: p.read_bytes() for p in first.cache_dir.rglob("*.json")}
    path, document = write_category_correction(base, first)
    record = document["corrections"][0]
    if problem == "variant":
        document["variant_key"] = contexts[1].variant_key
    elif problem == "url":
        document["source_url"] = contexts[1].source_url
    elif problem == "pdf":
        document["pdf_sha256"] = "0" * 64
    elif problem == "old":
        record["old"] = True
    elif problem == "label":
        record["source_label"] = "B999"
    elif problem == "field":
        record["field"] = "answer"
    elif problem == "duplicate":
        document["corrections"].append(dict(record))
    elif problem == "evidence":
        record["evidence"] = []
    elif problem == "method":
        record["method"] = ""
    elif problem == "parent_topic":
        record["new"] = 10
    elif problem == "unknown_exam":
        record.update(field="exams_id", new=999)
    elif problem == "placeholder":
        record["new"] = "нет подходящей темы"
    elif problem == "existing_topic_name":
        record["new"] = "Проценты"
    path.write_text("{" if problem == "invalid_json" else json.dumps(document), encoding="utf-8")

    report = run_batch(base, [first], client=forbid_model(FakeClient()))
    state = report["variants"][0]
    assert report["status"] == "FAILED" and "ValueError" in state["error"]
    assert state["stage"] == "CLASSIFICATION" and not state["result_current_attempt"]
    assert (report["ocr_calls"], report["classification_calls"]) == (0, 0)
    assert (first.output_dir / "tasks.xlsx").read_bytes() == previous_excel
    assert first.archive_path.read_bytes() == previous_zip
    assert {p.name: p.read_bytes() for p in first.cache_dir.rglob("*.json")} == original_cache


def test_correction_provenance_change_invalidates_content_review_without_paid_calls(project):
    base, contexts, _ = project
    first = contexts[0]
    run_batch(base, [first], client=FakeClient())
    path, document = write_category_correction(base, first)
    corrected = run_batch(base, [first], client=forbid_model(FakeClient()))
    fingerprint = corrected["variants"][0]["review_fingerprint"]
    (first.debug_dir / "content_review.json").write_text(json.dumps({
        "fingerprint": fingerprint, "approved": True, "findings": [],
    }), encoding="utf-8")
    document["corrections"][0]["reason"] += "; уточнено доказательство"
    path.write_text(json.dumps(document), encoding="utf-8")

    repeated = run_batch(base, [first], client=forbid_model(FakeClient()))
    state = repeated["variants"][0]
    assert state["review_fingerprint"] != fingerprint
    assert state["content_reviewed"] is False
    assert repeated["ocr_calls"] == repeated["classification_calls"] == 0


def test_condition_corrections_preserve_paid_cache_and_other_variant(project):
    base, contexts, _ = project
    run_batch(base, contexts, client=FakeClient())
    first, second = contexts
    cached = {str(p.relative_to(first.cache_dir)): p.read_bytes()
              for p in first.cache_dir.rglob('*.json')}
    other_excel = (second.output_dir / 'tasks.xlsx').read_bytes()
    other_zip = second.archive_path.read_bytes()
    initial = json.loads((first.cache_dir / 'conditions.json').read_text())['tasks'][0]['condition']
    path, document = write_category_correction(base, first)
    document['version'] = 2
    document['corrections'].append({
        'source_label': 'B1', 'field': 'condition', 'old': initial,
        'new': initial.replace('$x^2$', r'$\sqrt{x}-1$'),
        'reason': 'Корень по исходному PDF', 'evidence': ['source PDF page 1'],
        'method': 'manual-source-review',
    })
    path.write_text(json.dumps(document), encoding='utf-8')
    report = run_batch(base, [first], client=forbid_model(FakeClient()))
    state = report['variants'][0]
    assert (report['ocr_calls'], report['classification_calls']) == (0, 0)
    rows, _ = read_workbook(first)
    assert r'\sqrt{x}-1' in rows[1][1] and rows[1][6] == 12
    assert {str(p.relative_to(first.cache_dir)): p.read_bytes()
            for p in first.cache_dir.rglob('*.json')} == cached
    assert (second.output_dir / 'tasks.xlsx').read_bytes() == other_excel
    assert second.archive_path.read_bytes() == other_zip
    assert state['corrections']['records'] == document['corrections']
    assert not state['content_reviewed']


@pytest.mark.parametrize('problem', ['old', 'empty', 'non_string', 'version_one', 'answer'])
def test_invalid_condition_corrections_are_atomic(project, problem):
    from alexlarin_parser.corrections import apply_category_corrections
    base, contexts, _ = project
    context = contexts[0]
    tasks = [{'task_num': 1, 'source_label': 'B1', 'condition': r'$\sqrt{x}-1$',
              'exams_id': 11, 'topics_id': 11}]
    original = copy.deepcopy(tasks)
    path, document = write_category_correction(base, context)
    document['version'] = 2
    record = dict(source_label='B1', field='condition', old=tasks[0]['condition'],
                  new=r'$\sqrt{x-1}$', reason='Source verified',
                  evidence=['PDF page 1'], method='manual-source-review')
    document['corrections'].append(record)
    if problem == 'old': record['old'] += ' stale'
    elif problem == 'empty': record['new'] = '  '
    elif problem == 'non_string': record['new'] = 7
    elif problem == 'version_one': document['version'] = 1
    elif problem == 'answer': record['field'] = 'answer'
    path.write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(ValueError):
        apply_category_corrections(context, tasks, pdf_path=context.pdf_path,
                                   exams_path=base/'data/exams.csv', topics_path=base/'data/topics.csv')
    assert tasks == original


def test_published_answer_fills_missing_answer_without_model_calls(project):
    base, contexts, _ = project
    first, second = contexts
    class MissingAnswerClient(FakeClient):
        def request_json(self, *args, **kwargs):
            result = super().request_json(*args, **kwargs)
            if kwargs.get('stage') == 'ocr':
                result['answers'] = {}
            return result
    # The fixture's HTML provides B1; C1 has no HTML answer.
    run_batch(base, contexts, client=MissingAnswerClient())
    cached = {str(p.relative_to(first.cache_dir)): p.read_bytes()
              for p in first.cache_dir.rglob('*.json')}
    other_zip = second.archive_path.read_bytes()
    path, document = write_category_correction(base, first)
    document['version'] = 3
    document['corrections'].append(dict(source_label='C1', field='answer', old='',
        new='а) да; б) нет.', reason='Перенос опубликованного ответа для того же условия',
        evidence=['https://example.org/published-answer', 'local verified source excerpt'],
        method='published-source-transcription'))
    path.write_text(json.dumps(document), encoding='utf-8')
    report = run_batch(base, [first], client=forbid_model(FakeClient()))
    state = report['variants'][0]
    assert report['ocr_calls'] == report['classification_calls'] == 0
    assert not state['missing_answers'] and state['answers_count'] == 2
    assert not any('опубликованный ответ отсутствует' in w for w in state['warnings'])
    rows, _ = read_workbook(first)
    assert rows[2][4] == 'а) да;<br>б) нет.' and rows[2][3] is None
    assert {str(p.relative_to(first.cache_dir)): p.read_bytes()
            for p in first.cache_dir.rglob('*.json')} == cached
    assert second.archive_path.read_bytes() == other_zip
    assert state['corrections']['records'] == document['corrections']


@pytest.mark.parametrize('problem', ['old', 'empty', 'non_string', 'method', 'url', 'version_two'])
def test_invalid_published_answer_correction_is_atomic(project, problem):
    from alexlarin_parser.corrections import apply_task_corrections
    base, contexts, _ = project
    context = contexts[0]
    tasks = [dict(task_num=1, source_label='B1', condition='Condition', answer='',
                  exams_id=11, topics_id=11)]
    path, document = write_category_correction(base, context)
    document['version'] = 3
    record = dict(source_label='B1', field='answer', old='', new='Published answer',
        reason='Same source condition', evidence=['https://example.org/answer'],
        method='published-source-transcription')
    document['corrections'].append(record)
    if problem == 'old':
        tasks[0]['answer'] = record['old'] = 'Existing published answer'
    elif problem == 'empty': record['new'] = ' '
    elif problem == 'non_string': record['new'] = 1
    elif problem == 'method': record['method'] = 'model-solution'
    elif problem == 'url': record['evidence'] = ['no source URL']
    elif problem == 'version_two': document['version'] = 2
    original = copy.deepcopy(tasks)
    path.write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(ValueError):
        apply_task_corrections(context, tasks, pdf_path=context.pdf_path,
            exams_path=base/'data/exams.csv', topics_path=base/'data/topics.csv')
    assert tasks == original


def test_version_four_corrects_published_answer_and_preserves_cache(project):
    base, contexts, _ = project
    first, second = contexts
    run_batch(base, contexts, client=FakeClient())
    cached = {str(p.relative_to(first.cache_dir)): p.read_bytes()
              for p in first.cache_dir.rglob('*.json')}
    other = second.archive_path.read_bytes()
    path, document = write_category_correction(base, first)
    document['version'] = 4
    document['corrections'].append(dict(source_label='C1', field='answer', old=r'$\sqrt{2}$',
        new=r'$\sqrt{3}$', reason='Корень 3 в опубликованной PNG',
        evidence=['https://example.org/answers.png', 'source PNG hash and review'],
        method='published-source-transcription'))
    path.write_text(json.dumps(document), encoding='utf-8')
    report = run_batch(base, [first], client=forbid_model(FakeClient()))
    assert report['ocr_calls'] == report['classification_calls'] == 0
    assert read_workbook(first)[0][2][4] == r'$\sqrt{3}$'
    assert {str(p.relative_to(first.cache_dir)): p.read_bytes()
            for p in first.cache_dir.rglob('*.json')} == cached
    assert second.archive_path.read_bytes() == other
    assert report['variants'][0]['corrections']['records'] == document['corrections']


@pytest.mark.parametrize('problem', [None, 'unapproved', 'fingerprint', 'evidence', 'unknown', 'duplicate', 'not_page_warning', 'source_warning'])
def test_ocr_warning_resolution_requires_matching_review_and_evidence(project, problem):
    base, contexts, source_by_key = project
    first = contexts[0]
    if problem == 'source_warning':
        source_by_key[first.variant_key]['warnings'] = ['Страница 1: Source warning, not OCR']
    class IssueClient(FakeClient):
        def request_json(self, *args, **kwargs):
            result = super().request_json(*args, **kwargs)
            if kwargs.get('stage') == 'ocr':
                result['issues'] = ['Сверить продолжение условия по PDF']
            return result
    initial = run_batch(base, [first], client=IssueClient())['variants'][0]
    warnings = initial['ocr_warnings']
    assert len(warnings) == 2
    records = [dict(warning=w, reason='Продолжение сверено с полным PDF', evidence=['PDF page'])
               for w in warnings]
    review = dict(fingerprint=initial['review_fingerprint'], approved=True, findings=[],
                  resolved_warnings=records)
    if problem == 'unapproved': review['approved'] = False
    elif problem == 'fingerprint': review['fingerprint'] = 'stale'
    elif problem == 'evidence': records[-1]['evidence'] = []
    elif problem == 'unknown': records[-1]['warning'] = 'Страница 99: Unknown issue'
    elif problem == 'duplicate': records.append(records[0].copy())
    elif problem == 'not_page_warning': records[-1]['warning'] = initial['warnings'][-1]
    elif problem == 'source_warning': records[-1]['warning'] = 'Страница 1: Source warning, not OCR'
    (first.debug_dir/'content_review.json').write_text(json.dumps(review), encoding='utf-8')
    report = run_batch(base, [first], client=forbid_model(FakeClient()))
    state = report['variants'][0]
    assert report['ocr_calls'] == report['classification_calls'] == 0
    verification = json.loads((Path(state['debug_dir'])/'verification.json').read_text())
    if problem is None:
        assert state['status'] == 'SUCCESS' and not state['warnings']
        assert state['resolved_warnings'] == verification['resolved_warnings'] == records
    else:
        assert state['status'] == 'NEEDS_REVIEW'
        assert all(w in state['warnings'] for w in warnings)
        assert not state['resolved_warnings']
