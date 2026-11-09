"""Источники вариантов не разделяют локальные файлы и метки B/C."""

from pathlib import Path
import json

import pymupdf
import pytest
import requests

import downloader
import variant_sources
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


def test_continuation_label_is_one_main_identity_across_pages(tmp_path):
    context = VariantContext.from_url("https://alexlarin.net/ege/2013/trvar1.html", tmp_path)
    with pymupdf.open() as document:
        for label in ("B1.", "B1.", "C1."):
            page = document.new_page()
            page.insert_text((30, 50), label + " continued")
        result = inspect_pdf(document.tobytes(), context)
    assert result["source_labels"] == ["B1", "C1"]
    assert result["page_labels"] == [["B1"], ["B1"], ["C1"]]


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
    monkeypatch.setattr(variant_sources, "download_ranged_source", no_range)
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
    monkeypatch.setattr(variant_sources.requests, "get", get)
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
    monkeypatch.setattr(variant_sources.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="Range"):
        variant_sources.download_ranged_source(context, "https://example.org/source", chunk_size=8)
    assert context.download_count == 1
    assert not context.pdf_path.exists()
