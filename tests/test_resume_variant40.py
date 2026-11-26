"""Resume №40: настоящие этапы и артефакты в tmp_path, HTTP полностью подменён."""

import copy
import json
from pathlib import Path
import re
import shutil
import socket
from types import SimpleNamespace
from zipfile import ZipFile

from openpyxl import load_workbook
import pytest
import requests

from alexlarin_parser import cli, pipeline, provider
from alexlarin_parser.context import VariantContext
from alexlarin_parser.io import file_sha256
from alexlarin_parser.paths import PROJECT_ROOT


URL = "https://alexlarin.net/ege/2014/trvar40.html"


class Variant40Session:
    def __init__(self, case, failures=None):
        self.case = case
        self.failures = failures or {}
        self.calls = []
        self.closed = False

    def mount(self, *args):
        pass

    def close(self):
        self.closed = True

    def post(self, url, *, json: dict, **kwargs):
        messages = json["messages"]
        if isinstance(messages[0]["content"], list):
            page = int(re.search(r"'page_num': (\d+)", messages[0]["content"][0]["text"])[1])
            key = ("ocr", page)
            data = copy.deepcopy(self.case.pages[page])
        else:
            key = ("classification", None)
            data = copy.deepcopy(self.case.classification)
        self.calls.append(key)
        failure = self.failures.get(key)
        if isinstance(failure, BaseException):
            raise failure
        if failure == "partial":
            data["tasks"].pop()
        return completion(data, finish="length" if failure == "length" else "stop")


def completion(data, *, finish):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({
        "choices": [{"finish_reason": finish, "message": {"content": json.dumps(data)}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }).encode("utf-8")
    return response


@pytest.fixture
def variant40(tmp_path, monkeypatch):
    original = VariantContext.from_url(URL, PROJECT_ROOT)
    required = [original.source_dir, original.output_dir, original.archive_path,
                original.cache_dir / "classification.json", original.cache_dir / "conditions.json",
                *(original.cache_dir / "ocr" / f"page_{n:03d}.json" for n in (1, 2, 3))]
    if any(not path.exists() for path in required):
        pytest.skip("Нужны локальные исходники, сохранённые ответы и артефакты №40")
    original_paths = [path for root in (original.source_dir, original.cache_dir, original.output_dir)
                      for path in root.rglob("*") if path.is_file()] + [original.archive_path]
    original_hashes = {path: file_sha256(path) for path in original_paths}
    context = VariantContext.from_url(URL, tmp_path)
    shutil.copytree(original.source_dir, context.source_dir)
    shutil.copytree(original.output_dir, context.output_dir)
    context.archive_path.parent.mkdir(parents=True)
    shutil.copyfile(original.archive_path, context.archive_path)
    shutil.copytree(PROJECT_ROOT / "data", tmp_path / "data")
    tasks = json.loads((original.cache_dir / "conditions.json").read_text())["tasks"]
    cached_classification = json.loads((original.cache_dir / "classification.json").read_text())
    classification = cached_classification["response"]
    numbers = {task["task_num"]: n for n, task in enumerate(sorted(tasks, key=lambda t: t["condition"]), 1)}
    for task in classification["tasks"]:
        task["task_num"] = numbers[task["task_num"]]
    case = SimpleNamespace(
        base=tmp_path, context=context, classification=classification,
        pages={n: json.loads((original.cache_dir / "ocr" / f"page_{n:03d}.json").read_text())["response"]
               for n in (1, 2, 3)},
        categories={task["task_num"]: (task["exams_id"], task["topics_id"])
                    for task in cached_classification["tasks"]},
        labels=[task["source_label"] for task in tasks],
    )

    def forbid_network(*args, **kwargs):
        pytest.fail("Проверки resume не обращаются к сети")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    yield case
    assert all(file_sha256(path) == digest for path, digest in original_hashes.items())


def run(case, *, failures=None, fresh_ai=False, resume=None):
    session = Variant40Session(case, failures)
    client = provider.DeepSeekClient(provider.DeepSeekSettings(api_key="test-resume-only", dpi=72), session=session)
    contexts = pipeline.selected_contexts(case.base, resume=resume, local_only=True) if resume else [case.context]
    try:
        report = pipeline.run_batch(case.base, contexts, client=client, fresh_ai=fresh_ai,
                                    resume_from=resume, local_only=True)
    finally:
        client.close()
    return report, session


def artifacts(case):
    return {path: path.read_bytes() for path in case.context.output_dir.iterdir() if path.is_file()} | {
        case.context.archive_path: case.context.archive_path.read_bytes(),
    }


def assert_ready(case, report):
    state = report["variants"][0]
    # Структурный успех не выдаётся за новую содержательную сверку.
    assert report["status"] == "NEEDS_REVIEW" and state["error"] is None
    assert state["variant_key"] == "ege_2014_trvar40" and state["result_current_attempt"] is True
    assert (state["page_count"], state["task_count"], state["download_count"]) == (3, 20, 0)
    book = load_workbook(case.context.output_dir / "tasks.xlsx")
    try:
        rows = list(book["Tasks"].values)
        about = dict(zip(*list(book["about"].values)))
    finally:
        book.close()
    assert len(rows) == 21 and [row[0] for row in rows[1:]] == list(range(1, 21))
    assert [row[5:7] for row in rows[1:]] == [case.categories[n] for n in range(1, 21)]
    assert about["year"] == 2014 and about["source_url"] == URL
    images = {name for row in rows[1:] for name in (row[2] or "").splitlines()}
    with ZipFile(case.context.archive_path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == 1 + len(images)
        assert set(archive.namelist()) == {"tasks.xlsx", *images}
        assert archive.testzip() is None
        assert all(archive.read(name) == (case.context.output_dir / name).read_bytes() for name in archive.namelist())
    cache = Path(state["cache_dir"])
    assert len(list((cache / "ocr").glob("page_*.json"))) == 3
    tasks = json.loads((cache / "conditions.json").read_text())["tasks"]
    assert [task["source_label"] for task in tasks] == case.labels
    assert len({task["task_num"] for task in tasks}) == len(tasks) == 20
    assert len(json.loads((cache / "classification.json").read_text())["tasks"]) == 20


@pytest.mark.parametrize("stage,page,kind", [
    ("ocr", 2, "timeout"), ("ocr", 2, "partial"), ("ocr", 2, "length"),
    ("classification", None, "timeout"), ("classification", None, "partial"),
    ("classification", None, "length"),
])
def test_resume40_repeats_only_unfinished_model_work(variant40, stage, page, kind):
    case = variant40
    before = artifacts(case)
    failure = requests.Timeout("искусственный сбой") if kind == "timeout" else kind
    failed, first_session = run(case, failures={(stage, page): failure})
    state = failed["variants"][0]
    assert failed["status"] == "FAILED" and state["stage"] == stage.upper()
    assert state["result_current_attempt"] is False and artifacts(case) == before
    cache = Path(state["cache_dir"])
    ready_files = list((cache / "ocr").glob("page_*.json"))
    assert len(ready_files) == (1 if stage == "ocr" else 3)
    assert not (cache / "classification.json").exists()
    if stage == "ocr":
        assert not (cache / "conditions.json").exists()
    else:
        assert (cache / "conditions.json").is_file()
    ready_bytes = {path: path.read_bytes() for path in ready_files}
    manifest_before = Path(failed["manifest_path"]).read_bytes()
    resumed, session = run(case, resume=failed["manifest_path"])
    expected = [("ocr", 2), ("ocr", 3), ("classification", None)] if stage == "ocr" else [("classification", None)]
    assert session.calls == expected and first_session.closed and session.closed
    assert (resumed["ocr_calls"], resumed["classification_calls"]) == ((2, 1) if stage == "ocr" else (0, 1))
    assert resumed["resume_from"] == failed["manifest_path"] and resumed["manifest_path"] != failed["manifest_path"]
    assert Path(failed["manifest_path"]).read_bytes() == manifest_before
    assert all(path.read_bytes() == data for path, data in ready_bytes.items())
    assert_ready(case, resumed)
    repeated, session = run(case, resume=resumed["manifest_path"])
    assert session.calls == [] and (repeated["ocr_calls"], repeated["classification_calls"]) == (0, 0)
    assert_ready(case, repeated)


def test_resume40_after_export_failure_reuses_checked_classification(variant40, monkeypatch):
    case = variant40
    before = artifacts(case)

    def fail_export(*args, **kwargs):
        raise OSError("искусственный сбой экспорта")

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "export_variant", fail_export)
        failed, _ = run(case)
    assert failed["variants"][0]["stage"] == "EXPORT" and failed["status"] == "FAILED"
    assert artifacts(case) == before
    classification = case.context.cache_dir / "classification.json"
    saved = classification.read_bytes()
    resumed, session = run(case, resume=failed["manifest_path"])
    assert not session.calls and classification.read_bytes() == saved
    assert_ready(case, resumed)


def test_resume40_continues_fresh_cache_without_using_normal_ready_cache(variant40):
    case = variant40
    normal, _ = run(case)
    normal_files = {path: path.read_bytes() for path in case.context.cache_dir.rglob("*.json")}
    failed, session = run(case, fresh_ai=True, failures={("ocr", 2): requests.Timeout("fresh прерван")})
    assert session.calls == [("ocr", 1), ("ocr", 2)]
    fresh = Path(failed["variants"][0]["cache_dir"])
    assert fresh.is_relative_to(case.context.cache_dir / "fresh")
    saved = (fresh / "ocr" / "page_001.json").read_bytes()
    resumed, session = run(case, resume=failed["manifest_path"])
    assert session.calls == [("ocr", 2), ("ocr", 3), ("classification", None)]
    assert resumed["variants"][0]["cache_dir"] == str(fresh)
    assert (fresh / "ocr" / "page_001.json").read_bytes() == saved
    assert all(path.read_bytes() == data for path, data in normal_files.items())
    assert_ready(case, resumed)
    assert Path(normal["manifest_path"]).exists()


@pytest.mark.parametrize("stage", ["ocr", "classification"])
@pytest.mark.parametrize("corruption", ["partial_json", "invalid_response"])
def test_resume40_rejects_damaged_local_and_content_cache(variant40, stage, corruption):
    case = variant40
    ready, _ = run(case)
    path = (case.context.cache_dir / "ocr" / "page_002.json" if stage == "ocr" else
            case.context.cache_dir / "classification.json")
    shared = list((case.base / "input" / "cache" / "model_responses" / stage).glob("*.json"))
    if stage == "ocr":
        shared = [p for p in shared if json.loads(p.read_text())["response"]["page_num"] == 2]
    assert len(shared) == 1
    for cached_path in [path, *shared]:
        if corruption == "partial_json":
            cached_path.write_text('{"identity":')
        else:
            cached = json.loads(cached_path.read_text())
            cached["response"]["tasks"].pop()
            if "response_sha256" in cached:
                # Совпадение хеша не заменяет проверку полноты ответа.
                cached["response_sha256"] = pipeline.fingerprint(cached["response"])
                cached["record_sha256"] = pipeline.fingerprint({
                    "response": cached["response"], "provenance": cached.get("provenance"),
                })
            cached_path.write_text(json.dumps(cached))
    resumed, session = run(case, resume=ready["manifest_path"])
    assert session.calls == [(stage, 2 if stage == "ocr" else None)]
    assert_ready(case, resumed)


@pytest.mark.parametrize("failure,exit_code", [(requests.Timeout("сбой CLI"), 1), (KeyboardInterrupt(), 130)])
def test_cli_resume40_after_timeout_or_interrupt(variant40, monkeypatch, failure, exit_code):
    case = variant40
    sessions = []
    now = [provider.time.time()]
    monkeypatch.setattr(provider.time, "time", lambda: now[0])

    def session_factory():
        session = Variant40Session(case, {("ocr", 2): failure} if not sessions else None)
        sessions.append(session)
        return session

    monkeypatch.setattr(cli, "BASE_DIR", case.base)
    monkeypatch.setattr(provider.os, "environ", {"DEEPSEEK_API_KEY": "test-resume-only", "OCR_DPI": "72"})
    monkeypatch.setattr(provider.requests, "Session", session_factory)
    assert cli.cli(["--url", URL, "--local-only"]) == exit_code
    manifest = next((case.base / "reports").glob("*/manifest.json"))
    failed = json.loads(manifest.read_text())
    assert failed["status"] == "FAILED" and failed["ocr_calls"] == 2
    assert (case.context.cache_dir / "ocr" / "page_001.json").exists()
    assert not (case.context.cache_dir / "ocr" / "page_002.json").exists()
    assert failed["variants"][0]["attempts"][-1]["status"] == ("FAILED" if exit_code == 1 else "INTERRUPTED")
    if isinstance(failure, requests.Timeout):
        now[0] += 61  # Серверная пауза сохраняется между клиентами; реального сна нет.
    assert cli.cli(["--resume", str(manifest), "--local-only"]) == 2
    assert sessions[1].calls == [("ocr", 2), ("ocr", 3), ("classification", None)]
    resumed_path = next(path for path in (case.base / "reports").glob("*/manifest.json") if path != manifest)
    resumed = json.loads(resumed_path.read_text())
    assert_ready(case, resumed)
    assert all(session.closed for session in sessions)
