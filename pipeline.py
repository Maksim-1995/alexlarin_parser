"""Последовательный запуск вариантов с явным состоянием и отчётом попытки."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

from classification import classify_variant, load_catalog
from pipeline_io import atomic_json, file_sha256, fingerprint, json_value, read_json
from provider import DeepSeekClient, DeepSeekSettings, PaidCallsBlocked, usage_summary
from variant_export import export_variant
from variant_ocr import recognize_variant
from variant_sources import VariantContext, load_catalogue, prepare_sources, select_variants


def parse_selection(value):
    numbers = []
    for part in value.split(","):
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2 or not all(p.isdigit() for p in bounds):
                raise ValueError("Диапазон: 1-5 либо 1,2,3,4,5")
            first, last = map(int, bounds)
            if first > last:
                raise ValueError("Диапазон должен возрастать")
            numbers.extend(range(first, last + 1))
        elif part.strip().isdigit():
            numbers.append(int(part.strip()))
        else:
            raise ValueError("Диапазон: 1-5 либо 1,2,3,4,5")
    if not numbers or any(n <= 0 or n > 551 for n in numbers) or len(set(numbers)) != len(numbers):
        raise ValueError("Нужны уникальные основные номера от 1 до 551")
    return numbers


def selected_contexts(base_dir, *, numbers=None, url=None, resume=None, local_only=False):
    if resume is not None:
        previous = read_json(resume)
        if not isinstance(previous, dict) or not isinstance(previous.get("variants"), list):
            raise ValueError("Не удалось прочитать манифест --resume")
        contexts = [VariantContext.from_url(v["source_url"], base_dir) for v in previous["variants"]]
        if not contexts or len({c.variant_key for c in contexts}) != len(contexts):
            raise ValueError("Манифест содержит пустой/повторный набор вариантов")
        return contexts
    if url:
        return [VariantContext.from_url(url, base_dir)]
    catalogue = load_catalogue(base_dir, local_only=local_only)
    return [VariantContext.from_url(row["source_url"], base_dir)
            for row in select_variants(numbers or [1, 2, 3, 4, 5], catalogue)]


def plan_run(base_dir, contexts):
    plan = {
        "mode": "PLAN", "paid_calls": 0,
        "variants": [{"variant_key": c.variant_key, "number": c.number, "year": c.year,
                      "source_url": c.source_url, "source_dir": str(c.source_dir),
                      "output_dir": str(c.output_dir), "archive_path": str(c.archive_path)}
                     for c in contexts],
    }
    return plan


def _result_snapshot(context):
    excel = context.output_dir / "tasks.xlsx"
    return {"excel_path": str(excel), "archive_path": str(context.archive_path),
            "excel_exists": excel.is_file(), "archive_exists": context.archive_path.is_file(),
            "excel_sha256": file_sha256(excel) if excel.is_file() else None,
            "archive_sha256": file_sha256(context.archive_path) if context.archive_path.is_file() else None}


def run_batch(base_dir, contexts, *, client=None, fresh_ai=False, local_only=False,
              resume_from=None):
    base_dir = Path(base_dir)
    if fresh_ai and resume_from:
        raise ValueError("--fresh-ai создаёт новую попытку; для продолжения прежней используйте только --resume")
    resumed = {}
    if resume_from:
        old_manifest = read_json(resume_from)
        if not isinstance(old_manifest, dict) or not isinstance(old_manifest.get("variants"), list):
            raise ValueError("Не удалось прочитать манифест --resume")
        resumed = {v["variant_key"]: v for v in old_manifest["variants"]}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    report_dir = base_dir / "reports" / run_id
    manifest_path = report_dir / "manifest.json"
    report = {
        "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
        "fresh_ai": fresh_ai, "resume_from": str(resume_from) if resume_from else None,
        "provider": "deepseek-direct", "status": "RUNNING", "variants": [],
    }
    for c in contexts:
        report["variants"].append({
            "variant_key": c.variant_key, "number": c.number, "code": c.code,
            "year": c.year, "source_url": c.source_url, "stage": "PENDING", "status": "PENDING",
            "page_count": None, "task_count": None, "image_count": None,
            "missing_answers": [], "warnings": [], "error": None, "result": None,
            "previous_result": _result_snapshot(c), "result_current_attempt": False,
            "ocr_calls": 0, "classification_calls": 0, "download_count": 0,
            "stage_seconds": {}, "seconds": 0, "usage": usage_summary([]),
        })
    atomic_json(manifest_path, report)
    owned_client = client is None
    client = client or DeepSeekClient(DeepSeekSettings.from_env(base_dir))
    blocked = None
    batch_started = time.monotonic()

    def save():
        atomic_json(manifest_path, report)

    # Справочники проверяем до любых платных вызовов, включая OCR.
    try:
        load_catalog(base_dir / "data" / "exams.csv", name_column="name_of_Exam",
                     parent_column="parent_of_exam", root_id=2)
        load_catalog(base_dir / "data" / "topics.csv", name_column="name", parent_column="parent", root_id=1)
    except Exception as error:
        blocked = f"Справочники не прошли проверку: {error}"

    try:
        for context, state in zip(contexts, report["variants"]):
            started, attempt_offset = time.monotonic(), len(client.attempts)
            ocr_before, classification_before = client.counter.ocr, client.counter.classification
            downloads_before = context.download_count
            sources = None
            # Debug текущей попытки остаётся внутри variant debug, с уникальным run_id.
            # Контекст источников остаётся стабильным; API пишет в proxy ниже.
            run_debug = context.debug_dir / "runs" / run_id
            run_debug.mkdir(parents=True, exist_ok=True)
            state["debug_dir"] = str(run_debug)
            cache_dir = context.cache_dir / "fresh" / run_id if fresh_ai else None
            previous_state = resumed.get(context.variant_key)
            if previous_state and previous_state.get("cache_dir"):
                candidate = Path(previous_state["cache_dir"]).resolve()
                if previous_state.get("source_url") != context.source_url or not candidate.is_relative_to(context.cache_dir.resolve()):
                    raise ValueError("Манифест --resume содержит кэш другого варианта")
                cache_dir = candidate if candidate != context.cache_dir.resolve() else None
            state["cache_dir"] = str(cache_dir or context.cache_dir)

            @contextmanager
            def stage(name):
                state["stage"] = name
                save()
                stage_started = time.monotonic()
                try:
                    yield
                finally:
                    state["stage_seconds"][name] = round(time.monotonic() - stage_started, 3)

            try:
                if blocked:
                    raise PaidCallsBlocked(blocked)
                with stage("SOURCES"):
                    sources = prepare_sources(context, local_only=local_only)
                    state["page_count"] = sources["page_count"]
                    state["source_labels"] = sources["source_labels"]
                    state["source_urls"] = context.source_urls
                    state["warnings"].extend(sources.get("warnings", []))
                # Только debug переопределяется; идентичность и пути данных прежние.
                attempt_context = AttemptContext(context, run_debug)
                with stage("OCR"):
                    recognized = recognize_variant(attempt_context, sources, client, cache_dir=cache_dir,
                                                   allow_legacy=not fresh_ai)
                    state["ocr_provenance"] = recognized.get("provenance")
                    state["warnings"].extend(recognized.get("warnings", []))
                    tasks = recognized["tasks"]
                    state["task_count"] = len(tasks)
                    answers = dict(sources.get("html_answers", {}))
                    # Приоритет опубликованного HTML/JS, противоречие явно видно.
                    for label, answer in {**sources.get("extra_answers", {}), **recognized["answers"]}.items():
                        if label in answers and answers[label] != answer:
                            state["warnings"].append(f"Ответ {label}: источники расходятся, выбран HTML/JS")
                        else:
                            answers[label] = answer
                    unexpected = sorted(set(answers) - {t["source_label"] for t in tasks})
                    if unexpected:
                        state["warnings"].append(f"Ответы без соответствующих условий: {unexpected}")
                    for task in tasks:
                        task["variant_key"] = context.variant_key
                        task["answer"] = answers.get(task["source_label"], "")
                        task["solution"] = ""
                    state["missing_answers"] = [t["source_label"] for t in tasks if not t["answer"].strip()]
                    state["answers_count"] = len(tasks) - len(state["missing_answers"])
                with stage("CLASSIFICATION"):
                    classified = classify_variant(attempt_context, tasks, client,
                                                  exams_path=base_dir / "data" / "exams.csv",
                                                  topics_path=base_dir / "data" / "topics.csv", cache_dir=cache_dir)
                    for task in tasks:
                        task.update(classified[task["task_num"]])
                with stage("EXPORT"):
                    result = export_variant(attempt_context, tasks, sources["metadata"], pdf_path=sources["pdf_path"])
                    state["warnings"].extend(result.get("warnings", []))
                    state["result"] = json_value(result)
                    state["image_count"] = result["image_count"]
                    state["result_current_attempt"] = True
                state["review_fingerprint"] = fingerprint({
                    "variant_key": context.variant_key, "pdf_sha256": file_sha256(sources["pdf_path"]),
                    "tasks": tasks, "metadata": sources["metadata"],
                })
                review = read_json(context.debug_dir / "content_review.json")
                state["content_reviewed"] = bool(
                    isinstance(review, dict) and review.get("approved") is True
                    and review.get("fingerprint") == state["review_fingerprint"]
                )
                if not state["content_reviewed"]:
                    state["warnings"].append("Требуется содержательная сверка формул, ответов, рисунков и тем по источникам")
                state["status"] = "NEEDS_REVIEW" if state["warnings"] or state["missing_answers"] else "SUCCESS"
                state["stage"] = "COMPLETE"
                atomic_json(run_debug / "verification.json", {
                    "variant_key": context.variant_key, "source_url": context.source_url,
                    "source_labels": sources["source_labels"], "page_labels": sources["page_labels"],
                    "tasks": tasks, "missing_answers": state["missing_answers"],
                    "warnings": state["warnings"], "review_fingerprint": state["review_fingerprint"],
                    "result": state["result"], "status": state["status"],
                })
            except PaidCallsBlocked as error:
                blocked = str(error)
                state["status"], state["error"] = "FAILED", blocked
            except Exception as error:
                state["status"], state["error"] = "FAILED", f"{type(error).__name__}: {error}"
            except KeyboardInterrupt:
                state["status"], state["error"] = "FAILED", "Запуск прерван пользователем"
                raise
            finally:
                attempts = client.attempts[attempt_offset:]
                state["attempts"] = attempts
                state["usage"] = usage_summary(attempts)
                state["ocr_calls"] = client.counter.ocr - ocr_before
                state["classification_calls"] = client.counter.classification - classification_before
                state["download_count"] = context.download_count - downloads_before
                state["seconds"] = round(time.monotonic() - started, 3)
                atomic_json(report_dir / f"{context.variant_key}.json", state)
                save()
                print(f"{context.variant_key}: {state['status']}; OCR={state['ocr_calls']}, "
                      f"классификация={state['classification_calls']}, {state['seconds']} с", flush=True)
                if state["error"]:
                    print(state["error"], flush=True)
    finally:
        statuses = {v["status"] for v in report["variants"]}
        report["status"] = "FAILED" if statuses & {"FAILED", "PENDING"} else (
            "NEEDS_REVIEW" if "NEEDS_REVIEW" in statuses else "SUCCESS")
        report["seconds"] = round(time.monotonic() - batch_started, 3)
        report["ocr_calls"] = sum(v["ocr_calls"] for v in report["variants"])
        report["classification_calls"] = sum(v["classification_calls"] for v in report["variants"])
        report["usage"] = usage_summary([a for v in report["variants"] for a in v.get("attempts", [])])
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        if owned_client:
            client.close()
    print(f"Отчёт запуска: {manifest_path}", flush=True)
    return {**report, "manifest_path": str(manifest_path)}


class AttemptContext:
    """Представление контекста с debug текущей попытки без изменения исходного."""
    def __init__(self, context, debug_dir):
        self._context = context
        self.debug_dir = debug_dir

    def __getattr__(self, name):
        return getattr(self._context, name)
