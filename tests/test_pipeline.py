"""Пакетный поток с настоящими OCR-кэшами/Excel/ZIP и подменой платного API."""

import copy
import csv
import json
from pathlib import Path
import re
from zipfile import ZipFile

import pymupdf
import pytest
from openpyxl import load_workbook

from ai_requests import AIRequestCounter
import pipeline
from pipeline import plan_run, run_batch, selected_contexts
from provider import DeepSeekSettings, PaidCallsBlocked
from variant_sources import VariantContext


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

    def request_json(self, messages, *, stage, debug_dir):
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
            assert payload["variant_key"] == key
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
