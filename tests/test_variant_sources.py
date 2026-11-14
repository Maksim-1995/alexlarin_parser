"""Источники вариантов не разделяют локальные файлы и метки B/C."""

from pathlib import Path
import json
from types import SimpleNamespace

import pymupdf
import pytest
import requests

import downloader
import pipeline_io
import variant_sources
from alexlarin_parser.sources import parsing as source_parsing, service as source_service
from variant_sources import (
    VariantContext, discover_catalog, select_variants, parse_variant_html,
    normalize_label, inspect_pdf, task_positions, prepare_sources,
)


def html(number=1, year=2013, *, png=True):
    return (f'''<html><title>Тренировочный\n вариант № {number} ЕГЭ-{year}</title>
    <a href="trvar{number}.pdf">PDF</a><img src="logo.png">
    <table><tr><td><table>
    <tr><td>Задание</td><td>Ответ</td></tr>
    <tr><td><b>В1</b></td><td>0,96</td></tr>
    <tr><td>С1</td><td>−2</td></tr>
    </table></td></tr></table>
    ''' + (f'<img src="trvar{number}.png">' if png else '') + '</html>').encode()


def pdf(labels=("B1", "B14", "C1", "C6")):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((30, 40), "18 tasks (obsolete introduction)")
        for index, label in enumerate(labels):
            page.insert_text((30, 80 + 35 * index), f"{label}. Condition")
        return document.tobytes()


def png():
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 20, 20), False)
    pixmap.clear_with(255)
    return pixmap.tobytes("png")


@pytest.mark.parametrize("source_manifest", [False, True])
def test_shared_atomic_json_preserves_existing_formats(tmp_path, source_manifest):
    value = {"source_label": "В1", "condition": r"$\frac{1}{2}$", "pages": [1, 2]}
    path = tmp_path / "nested" / "cache.json"
    write = variant_sources.atomic_json if source_manifest else pipeline_io.atomic_json

    write(path, value)

    expected = json.dumps(value, ensure_ascii=False, indent=2)
    if source_manifest:
        expected += "\n"
    assert path.read_bytes() == expected.encode("utf-8")
    assert list(path.parent.iterdir()) == [path]


def test_shared_atomic_json_preserves_previous_cache_when_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    path.write_bytes(b'{"checked": true}')
    previous = path.read_bytes()

    def fail_replace(pending, destination):
        assert destination == path
        assert json.loads(pending.read_text()) == {"checked": False}
        assert path.read_bytes() == previous
        raise OSError("Замена кэша недоступна")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="Замена кэша"):
        variant_sources.atomic_json(path, {"checked": False})

    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_context_keys_paths_and_additional_versions_are_isolated(tmp_path):
    one = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    another_year = VariantContext.from_url("https://alexlarin.net/ege/2026/trvar1.html", tmp_path)
    main = VariantContext.from_url("http://alexlarin.net/ege/2013/trvar4.html", tmp_path)
    extra = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar4_2.html", tmp_path)
    assert one.variant_key == "ege_2013_trvar1"
    assert len({one.variant_key, another_year.variant_key, main.variant_key, extra.variant_key}) == 4
    assert main.source_url.startswith("https://")
    assert one.pdf_path == tmp_path / "input/variants/ege_2013_trvar1/source/trvar1.pdf"
    assert one.output_dir != another_year.output_dir
    assert one.archive_path.name == "ege_2013_trvar1_result.zip"
    assert extra.code == "trvar4_2"
    urls = ["trvar4*.html", "trvar4_star.html", "trvar4+.html", "trvar4_plus.html"]
    assert len({VariantContext.from_url("https://alexlarin.net/ege/2013/" + code, tmp_path).variant_key
                for code in urls}) == 4


def test_catalog_uses_actual_hrefs_and_selects_main_four(tmp_path):
    catalogue = discover_catalog(b'''<a href="ege/2013/trvar4_2.html">4</a>*
        <a href="ege/2013/trvar4.html">4</a>
        <a href="ege/2014/trvar52.html">52</a>''')
    assert select_variants([4], catalogue)[0]["source_url"].endswith("/2013/trvar4.html")
    assert catalogue[2]["year"] == 2014
    with pytest.raises(ValueError, match="не найден"):
        select_variants([1], catalogue)
    with pytest.raises(ValueError, match="однозначно"):
        select_variants([4], [*catalogue, catalogue[1]])


@pytest.mark.parametrize("original, expected", [("В01", "B1"), ("b14.", "B14"), ("С1", "C1"), ("C6)", "C6"), ("13.", "13")])
def test_only_explicit_visual_letter_normalization(original, expected):
    assert normalize_label(original) == expected


@pytest.mark.parametrize("invalid", ["A1", "А)", "Б)", "В)", "1.2", "C0", "B-1", 1])
def test_subtasks_and_invalid_labels_are_not_main_tasks(invalid):
    with pytest.raises(ValueError):
        normalize_label(invalid)


def test_html_answers_preserve_letter_identity_and_decimal_comma(tmp_path):
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    result = parse_variant_html(html(), context)
    assert result["answers"] == {"B1": "0,96", "C1": "−2"}
    assert result["answer_labels"] == {"B1": "В1", "C1": "С1"}
    assert result["resources"]["images"] == ["https://alexlarin.net/ege/2013/trvar1.png"]
    assert result["resources"]["js"] == []
    with pytest.raises(ValueError, match="другому варианту"):
        parse_variant_html(html(542, 2026), context)
    with pytest.raises(ValueError, match="Год"):
        parse_variant_html(html(1, 2026), context)


def test_actual_pdf_labels_override_obsolete_instruction_and_do_not_merge_b1_c1(tmp_path):
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    result = inspect_pdf(pdf(), context)
    assert result["source_labels"] == ["B1", "B14", "C1", "C6"]
    assert result["page_count"] == 1
    assert result["page_labels"] == [["B1", "B14", "C1", "C6"]]
    assert [p["original_label"] for p in result["page_positions"][0]] == ["B1.", "B14.", "C1.", "C6."]


@pytest.mark.parametrize("dash", ["-", "‐", "‑", "‒", "–", "—", "−"])
@pytest.mark.parametrize("last_year", [2013, 2014])
def test_exam_year_range_uses_end_year_with_unicode_dashes(tmp_path, monkeypatch, dash, last_year):
    """PDF7–12 используют U+2010; год URL не подменяется при несовпадении."""
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar7.html", tmp_path)
    header = f"Единый\u00a0государственный\u00a0экзамен, 2012{dash}{last_year}\u00a0г.\nТренировочный\u00a0вариант\u00a0№\u00a07"
    page = SimpleNamespace(
        rect=pymupdf.Rect(0, 0, 595, 842), rotation_matrix=pymupdf.Matrix(1, 1),
        get_text=lambda kind=None: [] if kind == "words" else header,
    )
    class Document:
        def __len__(self):
            return 1
        def __iter__(self):
            return iter([page])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(source_parsing.pymupdf, "open", lambda **kwargs: Document())
    if last_year == 2013:
        result = inspect_pdf(b"%PDF-test", context)
        assert not any("год" in warning for warning in result["warnings"])
        assert context.year == 2013
    else:
        with pytest.raises(ValueError, match="другому году"):
            inspect_pdf(b"%PDF-test", context)


def test_continuation_label_is_one_main_identity_across_pages(tmp_path):
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    with pymupdf.open() as document:
        for label in ("B1.", "B1.", "C1."):
            page = document.new_page()
            page.insert_text((30, 50), label + " continued")
        result = inspect_pdf(document.tobytes(), context)
    assert result["source_labels"] == ["B1", "C1"]
    assert result["page_labels"] == [["B1"], [], ["C1"]]


def test_split_label_across_pdf_blocks_uses_visual_adjacency_and_ignores_formula_numbers():
    from types import SimpleNamespace
    words = [
        (30, 50, 37, 60, "B", 2, 0, 0),
        (200, 80, 220, 90, "729.", 3, 0, 0),
        (37, 50, 42, 60, "8", 4, 0, 0),
        (30, 20, 43, 30, "B3", 5, 0, 0),
    ]
    page = SimpleNamespace(get_text=lambda kind: words, rotation_matrix=pymupdf.Matrix(1, 1))
    positions = task_positions(page)
    assert [p["source_label"] for p in positions] == ["B3", "B8"]
    assert positions[1]["original_label"] == "B8"


def test_split_cyrillic_letter_in_sentence_with_year_is_not_task_label():
    """В PDF7 «В 2008 году» начинается в body-indent рядом с меткой B13."""
    words = [
        (30, 20, 44, 30, "B1", 1, 0, 0),
        (430, 200, 452, 214, "B13", 2, 0, 0),
        (465, 202, 472, 212, "В", 2, 1, 0),
        (475, 202, 495, 212, "2008", 2, 1, 1),
        (499, 202, 520, 212, "году", 2, 1, 2),
        (430, 300, 437, 310, "С", 3, 0, 0),
        (438, 300, 443, 310, "1", 4, 0, 0),
    ]
    page = SimpleNamespace(get_text=lambda kind: words, rotation_matrix=pymupdf.Matrix(1, 1))

    assert [item["source_label"] for item in task_positions(page)] == ["B1", "B13", "C1"]


def test_rotated_pdf_positions_match_rendered_page_coordinates():
    with pymupdf.open() as document:
        page = document.new_page(width=595, height=842)
        page.insert_text((40, 700), "B1. Condition")
        original = pymupdf.Rect(*page.get_text("words")[0][:4])
        page.set_rotation(90)
        position = task_positions(page)[0]
        expected = original * page.rotation_matrix
        assert position["x"] == pytest.approx(expected.x0)
        assert position["y"] == pytest.approx(expected.y0)
        assert 0 <= position["x"] < page.rect.width
        assert 0 <= position["y"] < page.rect.height


def test_sources_download_once_then_reuse_bound_manifest(tmp_path, monkeypatch):
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    files = {"html": html(), "pdf": pdf(), "png": png()}
    calls = []
    def download(url):
        calls.append(url)
        return files[url.rsplit(".", 1)[1]]
    monkeypatch.setattr(downloader, "download_bytes", download)
    result = prepare_sources(context)
    assert context.download_count == 3
    assert result["html_answers"]["B1"] == "0,96"
    assert result["answer_images"] == [context.source_dir / "trvar1.png"]
    manifest = json.loads((context.source_dir / "source_manifest.json").read_text())
    assert manifest["variant_key"] == context.variant_key
    assert manifest["files"]["trvar1.png"]["url"].endswith("/2013/trvar1.png")
    monkeypatch.setattr(downloader, "download_bytes", lambda url: pytest.fail("Повторная сеть запрещена"))
    resumed = VariantContext.from_url(context.source_url, tmp_path)
    prepare_sources(resumed, local_only=True)
    assert resumed.download_count == 0


def test_missing_answer_image_is_warning_and_keeps_conditions(tmp_path, monkeypatch):
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    def download(url):
        if url.endswith("png"):
            raise requests.ConnectionError("нет PNG")
        return html() if url.endswith("html") else pdf()
    monkeypatch.setattr(downloader, "download_bytes", download)
    def no_range(context, url):
        raise requests.ConnectionError("нет PNG")
    monkeypatch.setattr(source_service, "download_ranged_source", no_range)
    result = prepare_sources(context)
    assert result["source_labels"] == ["B1", "B14", "C1", "C6"]
    assert not result["answer_images"]
    assert any("нет PNG" in warning for warning in result["warnings"])


def test_other_variant_manifest_cannot_bind_same_image_filename(tmp_path, monkeypatch):
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    monkeypatch.setattr(downloader, "download_bytes", lambda url: html() if url.endswith("html") else pdf() if url.endswith("pdf") else png())
    prepare_sources(context)
    manifest_path = context.source_dir / "source_manifest.json"
    data = json.loads(manifest_path.read_text())
    data["variant_key"] = "ege_2013_trvar2"
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="другому варианту"):
        prepare_sources(context)


def test_error_html_is_not_written_as_pdf(tmp_path, monkeypatch):
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    monkeypatch.setattr(downloader, "download_bytes", lambda url: html() if url.endswith("html") else b"<html>error</html>")
    with pytest.raises(ValueError, match="PDF"):
        prepare_sources(context)
    assert not context.pdf_path.exists()


def test_old_542_files_are_never_used_for_variant_one(tmp_path, monkeypatch):
    legacy = tmp_path / "input"
    legacy.mkdir()
    for suffix in ("html", "pdf", "png", "js"):
        (legacy / f"trvar542.{suffix}").write_bytes(b"do not use")
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    monkeypatch.setattr(downloader, "download_bytes", lambda url: pytest.fail("Нет разрешённого локального источника"))
    with pytest.raises(FileNotFoundError, match="trvar1.html"):
        prepare_sources(context, local_only=True)
    assert not context.html_path.exists()


def test_ranges_validate_offsets_total_and_count_each_request(tmp_path, monkeypatch):
    data = b"0123456789abcdefg"
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    class Response:
        status_code = 206
        def __init__(self, start, end):
            self.content = data[start:end + 1]
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}", "ETag": '"fixed"'}
        def raise_for_status(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    def get(url, headers, timeout):
        start, end = map(int, headers["Range"][6:].split("-"))
        return Response(start, min(end, len(data) - 1))
    monkeypatch.setattr(source_service.requests, "get", get)
    assert variant_sources.download_ranged_source(context, "https://example.org/source", chunk_size=8) == data
    assert context.download_count == 3


@pytest.mark.parametrize("header, body", [("bytes 1-8/17", b"12345678"), ("bytes 0-7/17", b"short"), ("bytes 0-7/*", b"01234567")])
def test_ranges_reject_invalid_response_without_partial_source(tmp_path, monkeypatch, header, body):
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    class Response:
        status_code = 206
        headers = {"Content-Range": header}
        content = body
        def raise_for_status(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(source_service.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="Range"):
        variant_sources.download_ranged_source(context, "https://example.org/source", chunk_size=8)
    assert context.download_count == 1
    assert not context.pdf_path.exists()
