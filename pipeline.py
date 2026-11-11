"""Последовательная обработка вариантов и координация её независимых этапов.

CLI выбирает контексты, а этот модуль управляет источниками, OCR,
классификацией и экспортом. Провайдер считает HTTP-попытки, отдельные этапы
владеют своими кэшами, манифест описывает только текущий запуск.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

from classification import classification_review_warnings, classify_variant, load_catalog
from pipeline_io import atomic_json, file_sha256, fingerprint, json_value, read_json
from provider import DeepSeekClient, DeepSeekSettings, PaidCallsBlocked, usage_summary
from variant_export import export_variant
from variant_ocr import recognize_variant
from variant_sources import VariantContext, load_catalogue, prepare_sources, select_variants


def parse_selection(value):
    """Разобрать список ``1,2,3`` или диапазон ``1-5`` с сохранением порядка.

    Raises:
        ValueError: Формат неверен, номера повторяются или выходят за 1–551.
    """
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


def _read_resume_manifest(path):
    """Прочитать общий формат манифеста для выбора и возобновления вариантов."""
    previous = read_json(path)
    if not isinstance(previous, dict) or not isinstance(previous.get("variants"), list):
        raise ValueError("Не удалось прочитать манифест --resume")
    return previous


def selected_contexts(base_dir, *, numbers=None, url=None, resume=None, local_only=False):
    """Получить контексты из манифеста, одного URL либо реальных ссылок архива.

    При resume сохранённые URL имеют приоритет: каталог архива не загружается.
    local_only ограничивает загрузку исходников, а не обращения к модели.
    """
    if resume is not None:
        previous = _read_resume_manifest(resume)
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
    """Вернуть JSON-совместимый план без создания клиента и обращения к AI.

    Пути уже определены контекстами. base_dir сохранён в публичной сигнатуре
    для совместимости с существующими вызывающими функциями.
    """
    return {
        "mode": "PLAN", "paid_calls": 0,
        "variants": [{"variant_key": c.variant_key, "number": c.number, "year": c.year,
                      "source_url": c.source_url, "source_dir": str(c.source_dir),
                      "output_dir": str(c.output_dir), "archive_path": str(c.archive_path)}
                     for c in contexts],
    }


def _result_snapshot(context):
    """Зафиксировать прежнюю выдачу, не принимая её за результат новой попытки."""
    excel = context.output_dir / "tasks.xlsx"
    return {"excel_path": str(excel), "archive_path": str(context.archive_path),
            "excel_exists": excel.is_file(), "archive_exists": context.archive_path.is_file(),
            "excel_sha256": file_sha256(excel) if excel.is_file() else None,
            "archive_sha256": file_sha256(context.archive_path) if context.archive_path.is_file() else None}


def _new_variant_state(context):
    """Создать отдельное изменяемое состояние: списки между вариантами не общие."""
    return {
        "variant_key": context.variant_key, "number": context.number, "code": context.code,
        "year": context.year, "source_url": context.source_url, "stage": "PENDING", "status": "PENDING",
        "page_count": None, "task_count": None, "image_count": None,
        "missing_answers": [], "warnings": [], "error": None, "result": None,
        "previous_result": _result_snapshot(context), "result_current_attempt": False,
        "ocr_calls": 0, "classification_calls": 0, "download_count": 0,
        "stage_seconds": {}, "seconds": 0, "usage": usage_summary([]),
    }


def _cache_directory(context, run_id, *, fresh_ai, previous_state):
    """Выбрать обычный, свежий или сохранённый кэш строго внутри варианта."""
    cache_dir = context.cache_dir / "fresh" / run_id if fresh_ai else None
    if previous_state and previous_state.get("cache_dir"):
        candidate = Path(previous_state["cache_dir"]).resolve()
        if (previous_state.get("source_url") != context.source_url
                or not candidate.is_relative_to(context.cache_dir.resolve())):
            raise ValueError("Манифест --resume содержит кэш другого варианта")
        cache_dir = candidate if candidate != context.cache_dir.resolve() else None
    # None сохраняет обычный режим, включая строго ограниченную legacy-совместимость.
    return cache_dir


@contextmanager
def _stage(state, name, save):
    """Сохранить начатый этап и измерить время даже при его ошибке."""
    state["stage"] = name
    save()
    started = time.monotonic()
    try:
        yield
    finally:
        state["stage_seconds"][name] = round(time.monotonic() - started, 3)


def _recognize_tasks(context, sources, state, client, *, cache_dir, fresh_ai):
    """Собрать OCR и опубликованные ответы по исходным меткам одного варианта."""
    recognized = recognize_variant(context, sources, client, cache_dir=cache_dir,
                                   allow_legacy=not fresh_ai)
    state["ocr_provenance"] = recognized.get("provenance")
    state["warnings"].extend(recognized.get("warnings", []))
    tasks = recognized["tasks"]
    state["task_count"] = len(tasks)
    answers = dict(sources.get("html_answers", {}))
    # HTML/JS имеет приоритет. Разногласия видны в отчёте, а не исправляются
    # дополнительным платным запросом или вычислением ответа.
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
    return tasks


def _classify_tasks(base_dir, context, tasks, state, client, *, cache_dir):
    """Назначить проверенные ID и отдельно отметить спорное содержание тем."""
    exams_path = base_dir / "data" / "exams.csv"
    topics_path = base_dir / "data" / "topics.csv"
    classified = classify_variant(context, tasks, client, exams_path=exams_path,
                                  topics_path=topics_path, cache_dir=cache_dir)
    for task in tasks:
        task.update(classified[task["task_num"]])
    state["warnings"].extend(classification_review_warnings(
        tasks, exams_path=exams_path, topics_path=topics_path,
    ))


def _apply_content_review(context, sources, tasks, result, state):
    """Применить только сверку, относящуюся к тем же данным и оформлению.

    Структурный успех экспорта не подтверждает правильность формул. Старое
    approved не переносится на изменённые условия, рисунки или форматтеры.
    """
    state["review_fingerprint"] = fingerprint({
        "variant_key": context.variant_key, "pdf_sha256": file_sha256(sources["pdf_path"]),
        "tasks": tasks, "metadata": sources["metadata"],
        "image_hashes": {Path(p).name: file_sha256(p) for p in result["image_files"]},
        "format_code_sha256": {
            name: file_sha256(Path(__file__).parent / name) for name in
            ("variant_export.py", "exporter.py", "latex_format.py", "task_html.py")
        },
    })
    review = read_json(context.debug_dir / "content_review.json")
    matching = isinstance(review, dict) and review.get("fingerprint") == state["review_fingerprint"]
    if matching:
        findings = review.get("findings", [])
        if isinstance(findings, list) and all(isinstance(f, str) for f in findings):
            state["warnings"].extend(findings)
    state["content_reviewed"] = bool(matching and review.get("approved") is True)
    if not state["content_reviewed"]:
        state["warnings"].append("Требуется содержательная сверка формул, ответов, рисунков и тем по источникам")
    state["status"] = "NEEDS_REVIEW" if state["warnings"] or state["missing_answers"] else "SUCCESS"
    state["stage"] = "COMPLETE"


def _write_verification(context, sources, tasks, state):
    """Сохранить внутренние метки и замечания вне итогового Excel и ZIP."""
    atomic_json(context.debug_dir / "verification.json", {
        "variant_key": context.variant_key, "source_url": context.source_url,
        "source_labels": sources["source_labels"], "page_labels": sources["page_labels"],
        "tasks": tasks, "missing_answers": state["missing_answers"],
        "warnings": state["warnings"], "review_fingerprint": state["review_fingerprint"],
        "result": state["result"], "status": state["status"],
    })


def _execute_variant(base_dir, context, state, client, *, run_debug, cache_dir,
                     fresh_ai, local_only, save):
    """Выполнить этапы варианта; обработка ошибок и учёт попыток находятся выше."""
    with _stage(state, "SOURCES", save):
        sources = prepare_sources(context, local_only=local_only)
        state["page_count"] = sources["page_count"]
        state["source_labels"] = sources["source_labels"]
        state["source_urls"] = context.source_urls
        state["warnings"].extend(sources.get("warnings", []))
    attempt_context = AttemptContext(context, run_debug)
    with _stage(state, "OCR", save):
        tasks = _recognize_tasks(attempt_context, sources, state, client,
                                 cache_dir=cache_dir, fresh_ai=fresh_ai)
    with _stage(state, "CLASSIFICATION", save):
        _classify_tasks(base_dir, attempt_context, tasks, state, client, cache_dir=cache_dir)
    with _stage(state, "EXPORT", save):
        result = export_variant(attempt_context, tasks, sources["metadata"], pdf_path=sources["pdf_path"])
        state["warnings"].extend(result.get("warnings", []))
        state["result"] = json_value(result)
        state["image_count"] = result["image_count"]
        state["result_current_attempt"] = True
    # Сверка общая для варианта, а диагностические файлы отдельные у каждой попытки.
    _apply_content_review(context, sources, tasks, result, state)
    _write_verification(attempt_context, sources, tasks, state)


def _run_variant(base_dir, context, state, client, *, run_id, report_dir,
                 previous_state, fresh_ai, local_only, blocked, save):
    """Ограничить ошибку одним вариантом и вернуть общий блокер платных вызовов."""
    started, attempt_offset = time.monotonic(), len(client.attempts)
    ocr_before, classification_before = client.counter.ocr, client.counter.classification
    downloads_before = context.download_count
    run_debug = context.debug_dir / "runs" / run_id
    run_debug.mkdir(parents=True, exist_ok=True)
    state["debug_dir"] = str(run_debug)
    cache_dir = _cache_directory(context, run_id, fresh_ai=fresh_ai, previous_state=previous_state)
    state["cache_dir"] = str(cache_dir or context.cache_dir)
    try:
        if blocked:
            raise PaidCallsBlocked(blocked)
        _execute_variant(base_dir, context, state, client, run_debug=run_debug, cache_dir=cache_dir,
                         fresh_ai=fresh_ai, local_only=local_only, save=save)
    except PaidCallsBlocked as error:
        blocked = str(error)
        state["status"], state["error"] = "FAILED", blocked
    except Exception as error:
        state["status"], state["error"] = "FAILED", f"{type(error).__name__}: {error}"
    except KeyboardInterrupt:
        state["status"], state["error"] = "FAILED", "Запуск прерван пользователем"
        raise
    finally:
        # Срез включает timeout и отклонённые ответы, не только успешные результаты.
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
    return blocked


def _complete_report(report, started):
    """Свести сохранённые состояния, включая прерванные/не начатые варианты."""
    statuses = {v["status"] for v in report["variants"]}
    report["status"] = "FAILED" if statuses & {"FAILED", "PENDING"} else (
        "NEEDS_REVIEW" if "NEEDS_REVIEW" in statuses else "SUCCESS")
    report["seconds"] = round(time.monotonic() - started, 3)
    report["ocr_calls"] = sum(v["ocr_calls"] for v in report["variants"])
    report["classification_calls"] = sum(v["classification_calls"] for v in report["variants"])
    report["usage"] = usage_summary([a for v in report["variants"] for a in v.get("attempts", [])])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()


def run_batch(base_dir, contexts, *, client=None, fresh_ai=False, local_only=False,
              resume_from=None):
    """Последовательно обработать варианты и вернуть итоговый отчёт с его путём.

    Переданный client принадлежит вызывающему коду; созданный здесь клиент
    закрывается после запуска. fresh_ai изолирует AI-кэши, resume_from повторно
    использует сохранённые пути. local_only ограничивает только источники.
    Ошибки вариантов записываются в манифест; авторизация/баланс блокируют
    последующие платные попытки всего пакета. KeyboardInterrupt сохраняется
    в отчёте и передаётся CLI для корректного кода завершения.
    """
    base_dir = Path(base_dir)
    if fresh_ai and resume_from:
        raise ValueError("--fresh-ai создаёт новую попытку; для продолжения прежней используйте только --resume")
    resumed = {}
    if resume_from:
        resumed = {v["variant_key"]: v for v in _read_resume_manifest(resume_from)["variants"]}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    report_dir = base_dir / "reports" / run_id
    manifest_path = report_dir / "manifest.json"
    report = {
        "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
        "fresh_ai": fresh_ai, "resume_from": str(resume_from) if resume_from else None,
        "provider": "deepseek-direct", "status": "RUNNING",
        "variants": [_new_variant_state(context) for context in contexts],
    }

    def save():
        """Записать общий прогресс после смены этапа или завершения варианта."""
        atomic_json(manifest_path, report)

    save()
    owned_client = client is None
    client = client or DeepSeekClient(DeepSeekSettings.from_env(base_dir))
    blocked = None
    started = time.monotonic()
    # Справочники проверяем до любых платных вызовов, включая OCR.
    try:
        load_catalog(base_dir / "data" / "exams.csv", name_column="name_of_Exam",
                     parent_column="parent_of_exam", root_id=2)
        load_catalog(base_dir / "data" / "topics.csv", name_column="name", parent_column="parent", root_id=1)
    except Exception as error:
        blocked = f"Справочники не прошли проверку: {error}"
    try:
        for context, state in zip(contexts, report["variants"]):
            blocked = _run_variant(
                base_dir, context, state, client, run_id=run_id, report_dir=report_dir,
                previous_state=resumed.get(context.variant_key), fresh_ai=fresh_ai,
                local_only=local_only, blocked=blocked, save=save,
            )
    finally:
        _complete_report(report, started)
        save()
        if owned_client:
            client.close()
    print(f"Отчёт запуска: {manifest_path}", flush=True)
    return {**report, "manifest_path": str(manifest_path)}


class AttemptContext:
    """Контекст с отдельным debug текущего запуска и общими путями варианта."""

    def __init__(self, context, debug_dir):
        """Переопределить только debug, не меняя переданный VariantContext."""
        self._context = context
        self.debug_dir = debug_dir

    def __getattr__(self, name):
        """Передать остальные свойства исходному контексту без копирования данных."""
        return getattr(self._context, name)
