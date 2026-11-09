"""Постраничный OCR старых вариантов: реальные PDF, подменённый AI-клиент."""

import base64
import copy
import json
from types import SimpleNamespace

import pymupdf
import pytest

import variant_ocr
from variant_ocr import recognize_variant, render_scheme


def make_pdf(path, page_starts, *, note=""):
    """Текст содержит реальные начала, но не подменяет визуальный OCR формул."""
    with pymupdf.open() as document:
        for index, labels in enumerate(page_starts):
            page = document.new_page(width=400, height=max(500, 120 + 50 * len(labels)))
            if not labels:
                page.insert_text((25, 40), "Instructions: 18 tasks. " + note)
            for offset, label in enumerate(labels):
                printed = f"{label}." if label.isdigit() else label
                page.insert_text((25, 70 + 45 * offset), f"{printed} Task page {index + 1}. {note}")
        document.save(path)
    return path


def page_response(number, labels, *, answers=None, issues=None):
    return {
        "page_num": number,
        "tasks": [{"source_label": label, "condition": f"Условие {label}: $x^2 + 1$.",
                   "continuation": False} for label in labels],
        "answers": answers or {}, "issues": issues or [],
    }


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.model = "deepseek-flash"
        self.failure_on_page = None
        self.settings = SimpleNamespace(
            dpi=72,
            identity=lambda stage: {"provider": "deepseek-direct", "model": self.model,
                                    "thinking": "disabled", "stage": stage},
        )

    def request_json(self, messages, *, stage, debug_dir):
        content = messages[0]["content"]
        # Номер страницы передан явно; не вычисляем его из количества вызовов
        # (при возобновлении первой оплачиваемой может быть вторая страница).
        text = content[0]["text"]
        marker = "'page_num': "
        number = int(text.rsplit(marker, 1)[1].split(",", 1)[0])
        self.calls.append({"page_num": number, "messages": messages, "stage": stage, "debug_dir": debug_dir})
        if self.failure_on_page == number:
            raise TimeoutError("Тестовый тайм-аут чтения DeepSeek")
        return copy.deepcopy(self.responses[number])


@pytest.fixture
def context(tmp_path):
    return SimpleNamespace(
        base_dir=tmp_path,
        variant_key="ege_2013_trvar1", source_url="https://alexlarin.net/ege/2013/trvar1.html",
        cache_dir=tmp_path / "cache", debug_dir=tmp_path / "debug",
    )


@pytest.fixture
def sources(tmp_path):
    labels = [["B1", "B2"], ["C1", "C2"]]
    return {
        "pdf_path": make_pdf(tmp_path / "variant.pdf", labels),
        "page_labels": labels, "source_labels": ["B1", "B2", "C1", "C2"],
        "page_count": 2, "answer_images": [],
    }


@pytest.fixture
def client(sources):
    return FakeClient({number: page_response(number, labels)
                       for number, labels in enumerate(sources["page_labels"], 1)})


def recognize(context, sources, client, **kwargs):
    return recognize_variant(context, sources, client, allow_legacy=False, **kwargs)


def test_cold_pages_then_zero_calls_and_b1_c1_have_distinct_numbers(context, sources, client):
    result = recognize(context, sources, client)
    assert [(t["source_label"], t["task_num"]) for t in result["tasks"]] == [
        ("B1", 1), ("B2", 2), ("C1", 3), ("C2", 4),
    ]
    assert len(client.calls) == sources["page_count"]
    assert [call["stage"] for call in client.calls] == ["ocr", "ocr"]
    assert result["tasks"][0]["section"] == "B"
    assert result["tasks"][2]["section"] == "C"
    assert result["tasks"][0]["page_numbers"] == [1]
    cached_pages = sorted((context.cache_dir / "ocr").glob("page_*.json"))
    assert len(cached_pages) == 2

    client.failure_on_page = 1
    warm = recognize(context, sources, client)
    assert warm == result
    assert len(client.calls) == 2


def test_continuation_fragments_preserved_on_next_page(context, tmp_path):
    labels = [["B1"], ["C1"]]
    sources = {"pdf_path": make_pdf(tmp_path / "continued.pdf", labels),
               "source_labels": ["B1", "C1"], "page_labels": labels, "answer_images": []}
    second = page_response(2, ["C1"])
    second["tasks"].insert(0, {"source_label": "B1", "condition": "Б) Найдите $\\sqrt{2}$.", "continuation": True})
    client = FakeClient({1: page_response(1, ["B1"]), 2: second})
    result = recognize(context, sources, client)
    assert len(result["tasks"]) == 2
    first = result["tasks"][0]
    assert first["page_numbers"] == [1, 2]
    assert len(first["fragments"]) == 2
    assert first["condition"] == "Условие B1: $x^2 + 1$.\nБ) Найдите $\\sqrt{2}$."
    assert result["tasks"][1]["task_num"] == 2
    assert result["tasks"][1]["source_label"] == "C1"


def test_instruction_page_not_counted_as_eighteen_tasks(context, tmp_path):
    labels = [[], [f"B{n}" for n in range(1, 15)], [f"C{n}" for n in range(1, 7)]]
    sources = {"pdf_path": make_pdf(tmp_path / "twenty.pdf", labels), "page_labels": labels,
               "source_labels": [label for page in labels for label in page], "answer_images": []}
    client = FakeClient({n: page_response(n, page) for n, page in enumerate(labels, 1)})
    result = recognize(context, sources, client)
    assert len(result["tasks"]) == 20
    assert [task["task_num"] for task in result["tasks"]] == list(range(1, 21))
    assert result["tasks"][-1]["source_label"] == "C6"
    assert len(client.calls) == 3
    assert json.loads((context.cache_dir / "ocr" / "page_001.json").read_text())["response"]["tasks"] == []


def make_answer_image(path):
    with pymupdf.open() as document:
        page = document.new_page(width=100, height=40)
        page.insert_text((10, 20), "C1: sqrt(2)")
        path.write_bytes(page.get_pixmap().tobytes("png"))
    return path


def test_published_answers_attached_only_last_request_with_separate_caption(context, sources, client, tmp_path):
    image = make_answer_image(tmp_path / "published_answers.png")
    sources["answer_images"] = [image]
    client.responses[2]["answers"] = {"C1": "$\\sqrt{2}$", "C2": "1,5"}
    result = recognize(context, sources, client)
    assert result["answers"] == {"C1": "$\\sqrt{2}$", "C2": "1,5"}
    assert len(client.calls) == 2
    first_content = client.calls[0]["messages"][0]["content"]
    last_content = client.calls[1]["messages"][0]["content"]
    assert len([part for part in first_content if part["type"] == "image_url"]) == 1
    assert len([part for part in last_content if part["type"] == "image_url"]) == 2
    assert not any("ОПУБЛИКОВАННЫЕ ОТВЕТЫ (не условия)" in part.get("text", "") for part in first_content)
    assert any("ОПУБЛИКОВАННЫЕ ОТВЕТЫ (не условия)" in part.get("text", "") for part in last_content)
    answer_part = [part for part in last_content if part["type"] == "image_url"][-1]
    assert base64.b64decode(answer_part["image_url"]["url"].split(",", 1)[1]) == image.read_bytes()
    assert "Это OCR, задачи НЕ решай" in first_content[0]["text"]
    assert "не вычисляй его" in first_content[0]["text"]


@pytest.mark.parametrize("change", ["pdf", "variant", "url", "model", "prompt", "schema", "dpi"])
def test_incompatible_page_cache_never_accepted(context, sources, client, tmp_path, monkeypatch, change):
    recognize(context, sources, client)
    if change == "pdf":
        sources["pdf_path"] = make_pdf(tmp_path / "changed.pdf", sources["page_labels"], note="changed")
    elif change == "variant":
        context.variant_key = "ege_2013_trvar2"
    elif change == "url":
        context.source_url = "https://alexlarin.net/ege/2013/trvar2.html"
    elif change == "model":
        client.model = "new-model"
    elif change == "prompt":
        monkeypatch.setattr(variant_ocr, "OCR_PROMPT_VERSION", "changed")
    elif change == "schema":
        monkeypatch.setattr(variant_ocr, "OCR_SCHEMA_VERSION", 999)
    else:
        client.settings.dpi = 96
    recognize(context, sources, client)
    assert len(client.calls) == 4


def test_answer_image_hash_invalidates_last_page_only(context, sources, client, tmp_path):
    image = make_answer_image(tmp_path / "answers.png")
    sources["answer_images"] = [image]
    recognize(context, sources, client)
    with pymupdf.open() as document:
        page = document.new_page(width=110, height=40)
        page.insert_text((10, 20), "C1: 2")
        image.write_bytes(page.get_pixmap().tobytes("png"))
    recognize(context, sources, client)
    assert [call["page_num"] for call in client.calls] == [1, 2, 2]


def test_timeout_keeps_first_complete_page_then_resumes(context, sources, client):
    client.failure_on_page = 2
    with pytest.raises(TimeoutError):
        recognize(context, sources, client)
    assert (context.cache_dir / "ocr" / "page_001.json").is_file()
    assert not (context.cache_dir / "ocr" / "page_002.json").exists()
    assert not (context.cache_dir / "conditions.json").exists()
    client.failure_on_page = None
    result = recognize(context, sources, client)
    assert len(result["tasks"]) == 4
    assert [call["page_num"] for call in client.calls] == [1, 2, 2]


def test_failed_changed_pdf_keeps_previous_aggregate(context, sources, client, tmp_path):
    recognize(context, sources, client)
    aggregate = context.cache_dir / "conditions.json"
    previous = aggregate.read_bytes()
    sources["pdf_path"] = make_pdf(tmp_path / "changed.pdf", sources["page_labels"], note="changed")
    client.failure_on_page = 2
    with pytest.raises(TimeoutError):
        recognize(context, sources, client)
    assert aggregate.read_bytes() == previous
    assert [call["page_num"] for call in client.calls] == [1, 2, 1, 2]


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "unknown", "empty", "wrong_page", "invented_answer", "orphan_continuation"])
def test_bad_second_page_not_ready_cache_and_no_hidden_repair(context, sources, client, invalid):
    response = client.responses[2]
    if invalid == "missing":
        response["tasks"].pop()
    elif invalid == "duplicate":
        response["tasks"].append(copy.deepcopy(response["tasks"][0]))
    elif invalid == "unknown":
        response["tasks"][0]["source_label"] = "C7"
    elif invalid == "empty":
        response["tasks"][0]["condition"] = " "
    elif invalid == "wrong_page":
        response["page_num"] = 99
    elif invalid == "invented_answer":
        response["answers"] = {"C1": "42"}
    else:
        response["tasks"][0]["continuation"] = True
    with pytest.raises(ValueError):
        recognize(context, sources, client)
    assert [call["page_num"] for call in client.calls] == [1, 2]
    assert (context.cache_dir / "ocr" / "page_001.json").is_file()
    assert not (context.cache_dir / "ocr" / "page_002.json").exists()
    assert not (context.cache_dir / "conditions.json").exists()


def test_suspicious_text_saved_as_review_warning_without_guessing(context, sources, client):
    client.responses[1]["tasks"][0]["condition"] = "Найдите [НЕРАЗБОРЧИВО]."
    client.responses[1]["issues"] = ["Не читается показатель степени B1"]
    result = recognize(context, sources, client)
    assert result["tasks"][0]["condition"] == "Найдите [НЕРАЗБОРЧИВО]."
    assert any("нечитаемый" in warning for warning in result["warnings"])
    assert any("показатель степени" in warning for warning in result["warnings"])
    assert (context.cache_dir / "ocr" / "page_001.json").is_file()


def test_original_cyrillic_source_label_survives_cache_round_trip(context, sources, client):
    client.responses[1]["tasks"][0]["source_label"] = "В1"
    cold = recognize(context, sources, client)
    assert cold["tasks"][0]["source_label"] == "B1"
    assert cold["tasks"][0]["original_label"] == "В1"
    warm = recognize(context, sources, client)
    assert warm["tasks"][0]["original_label"] == "В1"
    assert warm == cold
    assert len(client.calls) == 2


def test_fresh_cache_root_does_not_overwrite_checked_cache(context, sources, client, tmp_path):
    recognize(context, sources, client)
    originals = {p: p.read_bytes() for p in context.cache_dir.rglob("*.json")}
    fresh = tmp_path / "measurement-cache"
    recognize(context, sources, client, cache_dir=fresh)
    assert len(client.calls) == 4
    assert len(list((fresh / "ocr").glob("*.json"))) == 2
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_layout_crops_only_when_pdf_has_two_real_columns(tmp_path):
    with pymupdf.open() as document:
        full = document.new_page(width=400, height=500)
        full.insert_text((20, 80), "B1. One column")
        full.insert_text((20, 160), "C1. Still one column")
        assert render_scheme(full, 72)["layout"] == "full_page"
        full.insert_text((220, 80), "C2. Right column")
        scheme = render_scheme(full, 72)
        assert scheme["layout"] == "two_columns"
        assert len(scheme["clips"]) == 2


def test_changed_catalogue_files_cannot_invalidate_ocr(context, sources, client, tmp_path):
    catalog = tmp_path / "topics.csv"
    catalog.write_text("id,name,parent\n1,Математика,0\n", encoding="utf8")
    recognize(context, sources, client)
    catalog.write_text("id,name,parent\n1,Математика,0\n2,Алгебра,1\n", encoding="utf8")
    recognize(context, sources, client)
    assert len(client.calls) == 2
