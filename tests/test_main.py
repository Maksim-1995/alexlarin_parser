"""CLI DeepSeek и сохранённые проверки чистых функций источников 542.

Постраничные интеграции нового потока находятся в test_pipeline.py и
 test_variant_ocr.py; обычные тесты не обращаются к платным API.
"""

import csv
import json
import shutil
import socket
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pymupdf
import pytest
from openpyxl import load_workbook

import classification
import downloader
import main
import recognizer
from export_metadata import ABOUT_FIELDS


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = (
    "trvar542.html", "trvar542.pdf", "trvar542.js", "trvar542.png",
    "trvar542_answers.json", "trvar542_conditions_gemini.json",
)
IMAGE_NAMES = {1: "image_82.jpeg", 3: "image_83.jpeg", 8: "image_198.png", 11: "image_200.png"}
EXAM_LABEL = "E | Вектор > Векторы и операции с ними"
TOPIC_LABEL = (
    "T | Геометрия > Планиметрия > Векторы > "
    "Умножение вектора на число. Скалярное произведение векторов"
)
NEW_TOPIC = "Первообразная и определённый интеграл"


@pytest.fixture
def local_project(tmp_path, monkeypatch):
    missing = [name for name in SOURCE_NAMES if not (PROJECT / "input" / name).is_file()]
    if missing:
        pytest.skip("Для проверки нужны сохранённые реальные исходники: " + ", ".join(missing))
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in SOURCE_NAMES:
        shutil.copyfile(PROJECT / "input" / name, input_dir / name)
    # Пользовательский кэш может быть частичным после прерывания OCR.
    # Полный legacy-кэш для этих тестов создаём только в tmp: реальные
    # условия сохраняем, отсутствующие дополняем явно тестовым текстом.
    cache_path = input_dir / "trvar542_conditions_gemini.json"
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    conditions = recognizer.validate_conditions(cached["tasks"], require_all=False)
    cached["tasks"] = [
        {"task_num": number, "condition": conditions.get(number, "Тестовое условие для проверки сборки.")}
        for number in range(1, 20)
    ]
    cache_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    shutil.copytree(PROJECT / "data", tmp_path / "data")
    monkeypatch.setattr(main, "BASE_DIR", tmp_path)
    monkeypatch.setattr(main, "INPUT_DIR", input_dir)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def network_forbidden(*args, **kwargs):
        pytest.fail("Сеть при локальной сборке запрещена")

    monkeypatch.setattr(socket.socket, "connect", network_forbidden)
    monkeypatch.setattr(downloader, "download_bytes", network_forbidden)
    monkeypatch.setattr(recognizer.genai, "Client", network_forbidden)
    return tmp_path


def test_plan_cli_never_constructs_provider(tmp_path, monkeypatch, capsys):
    import pipeline
    from variant_sources import VariantContext
    monkeypatch.setattr(main, "BASE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "DeepSeekClient", lambda *a, **kw: pytest.fail("План не вызывает AI"))
    url = "https://alexlarin.net/ege/2013/trvar1.html"
    assert main.cli(["--plan", "--url", url]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["paid_calls"] == 0
    assert plan["variants"][0]["variant_key"] == "ege_2013_trvar1"
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("status,code", [("SUCCESS",0),("NEEDS_REVIEW",2),("FAILED",1)])
def test_cli_routes_single_batch_and_resume_through_same_pipeline(tmp_path, monkeypatch, status, code):
    import pipeline
    from variant_sources import VariantContext
    seen = []
    contexts = [VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)]
    monkeypatch.setattr(pipeline, "selected_contexts", lambda *a, **kw: contexts)
    def run(base_dir, actual, **kwargs):
        seen.append((actual, kwargs))
        return {"status": status}
    monkeypatch.setattr(pipeline, "run_batch", run)
    assert main.cli(["--batch", "1-5", "--fresh-ai", "--local-only"]) == code
    assert seen[0][0] == contexts
    assert seen[0][1]["fresh_ai"] is True
    assert seen[0][1]["local_only"] is True


def test_cli_batch_size_guard_and_error_exit(tmp_path, monkeypatch, capsys):
    import pipeline
    contexts = [object()] * 6
    monkeypatch.setattr(pipeline, "selected_contexts", lambda *a, **kw: contexts)
    monkeypatch.setattr(pipeline, "run_batch", lambda *a, **kw: pytest.fail("Большой платный пакет запрещён"))
    assert main.cli(["--batch", "1-6"]) == 1
    assert "max-variants" in capsys.readouterr().err


@pytest.mark.parametrize("encoding", ["utf-8", "windows-1251"])
def test_html_uses_original_url_and_exact_variant(encoding):
    html = '''<html><title>Тренировочный вариант №542 ЕГЭ - 2026</title>
    <a href="other.pdf">Other</a><a href="trvar542.pdf">PDF</a>
    <script src="%74rvar542.js"></script><img src="trvar542.png"></html>'''
    assert main.parse_source_page(html.encode(encoding), main.DEFAULT_PAGE_URL) == {
        extension: f"https://alexlarin.net/ege/2026/trvar542.{extension}"
        for extension in ("pdf", "js", "png")
    }
    with pytest.raises(ValueError, match="нужного тренировочного варианта"):
        main.parse_source_page(html.replace("№542", "№541").encode(encoding), main.DEFAULT_PAGE_URL)
    with pytest.raises(ValueError, match="Год"):
        main.parse_source_page(html.replace("2026", "2025").encode(encoding), main.DEFAULT_PAGE_URL)
    with pytest.raises(ValueError, match="trvar542.pdf"):
        main.parse_source_page(html.replace("trvar542.pdf", "trvar541.pdf").encode(encoding), main.DEFAULT_PAGE_URL)


@pytest.mark.parametrize("reader", [main.read_js_source, main.validate_answer_image])
def test_error_page_cannot_replace_js_or_image(reader):
    with pytest.raises(ValueError):
        reader(b"<html>503 Service Unavailable</html>")


def test_corrupt_packed_js_is_rejected_with_clear_error():
    with pytest.raises(ValueError, match="JavaScript"):
        main.read_js_source(b"eval(function(p,a,c,k,e,r){broken")


def test_real_pdf_with_wrong_number_or_year_is_rejected(local_project):
    data = (local_project / "input" / "trvar542.pdf").read_bytes()
    with pytest.raises(ValueError, match="другому варианту"):
        main.read_pdf_source(data, "https://alexlarin.net/ege/2026/trvar541.html")
    with pytest.raises(ValueError, match="Год"):
        main.read_pdf_source(data, "https://alexlarin.net/ege/2025/trvar542.html")


@pytest.mark.parametrize("extension", ["pdf", "js", "png"])
@pytest.mark.parametrize("wrong_prefix", [
    "https://other.example/ege/2026/", "https://alexlarin.net/ege/2025/",
])
def test_html_rejects_resource_from_other_origin_or_year(extension, wrong_prefix):
    html = '''<html><title>Тренировочный вариант №542 ЕГЭ - 2026</title>
    <a href="trvar542.pdf">PDF</a><script src="trvar542.js"></script>
    <img src="trvar542.png"></html>'''
    wrong_html = html.replace(f'"trvar542.{extension}"', f'"{wrong_prefix}trvar542.{extension}"')
    with pytest.raises(ValueError, match=f"trvar542.{extension}"):
        main.parse_source_page(wrong_html.encode(), main.DEFAULT_PAGE_URL)


@pytest.fixture
def extra_answers():
    return {
        "source_url": "https://alexlarin.net/ege/2026/trvar542.png",
        "method": "manual_transcription",
        "answers": {str(number): "0,9940" if number == 13 else "$\\frac{2}{3}$" for number in range(13, 20)},
    }


def test_extra_answers_keep_original_strings_and_validate_source(tmp_path, extra_answers):
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(extra_answers), encoding="utf-8")
    result = main.read_extra_answers(path, extra_answers["source_url"])
    assert list(result["answers"]) == list(range(13, 20))
    assert result["answers"][13] == "0,9940"
    assert result["answers"][17] == "$\\frac{2}{3}$"
    assert result["method"] == "manual_transcription"
    with pytest.raises(ValueError, match="другой картинке"):
        main.read_extra_answers(path, "https://alexlarin.net/ege/2026/trvar541.png")


@pytest.mark.parametrize("corruption", [
    "missing", "invalid_json", "not_object", "missing_answers", "not_answers_object",
    "missing_method", "invalid_number", "empty_answer", "boolean_answer", "missing_task", "duplicate_task",
])
def test_extra_answers_fail_with_clear_error(tmp_path, extra_answers, corruption):
    path = tmp_path / "answers.json"
    image_url = extra_answers["source_url"]
    if corruption == "not_object":
        extra_answers = []
    elif corruption == "missing_answers":
        extra_answers.pop("answers")
    elif corruption == "not_answers_object":
        extra_answers["answers"] = []
    elif corruption == "missing_method":
        extra_answers.pop("method")
    elif corruption == "invalid_number":
        extra_answers["answers"]["thirteen"] = extra_answers["answers"].pop("13")
    elif corruption == "empty_answer":
        extra_answers["answers"]["13"] = " "
    elif corruption == "boolean_answer":
        extra_answers["answers"]["13"] = True
    elif corruption == "missing_task":
        extra_answers["answers"].pop("13")
    elif corruption == "duplicate_task":
        extra_answers["answers"]["013"] = "1"
    if corruption != "missing":
        path.write_text("broken JSON" if corruption == "invalid_json" else json.dumps(extra_answers), encoding="utf-8")
    with pytest.raises(ValueError, match="answers.json"):
        main.read_extra_answers(path, image_url)
