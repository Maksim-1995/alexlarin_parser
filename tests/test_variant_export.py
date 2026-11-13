"""Изоляция выдачи, старый контракт 542 и сохранность прежних результатов."""

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pymupdf
import pytest
from openpyxl import load_workbook

import variant_export
from export_metadata import ABOUT_FIELDS
from exporter import create_zip
from variant_export import export_variant, extract_variant_images, normalize_answer


def context(root, number):
    key = f"ege_2013_trvar{number}"
    return SimpleNamespace(
        variant_key=key, number=number, code=f"trvar{number}", year=2013,
        source_url=f"https://alexlarin.net/ege/2013/trvar{number}.html",
        output_dir=root / "output" / key, archive_path=root / "archives" / f"{key}_result.zip",
        debug_dir=root / "input" / "variants" / key / "debug",
    )


def picture(color):
    with pymupdf.open() as document:
        page = document.new_page(width=60, height=50)
        page.draw_rect((5, 5, 50, 40), color=color, fill=color)
        return page.get_pixmap().tobytes("png")


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=500)
        page.insert_text((30, 40), "B1. Text")
        page.insert_image((100, 65, 160, 115), stream=picture((1, 0, 0)))
        page.insert_image((200, 65, 260, 115), stream=picture((0, 0, 1)))
        page.insert_text((30, 200), "C1. Text")
        document.save(path)
    positions = {1: [{"source_label": "B1", "x": 30, "y": 29},
                     {"source_label": "C1", "x": 30, "y": 189}]}
    return path, positions


def tasks(mark):
    return [
        {"task_num": 1, "source_label": "B1", "condition": f"Вариант {mark}. Найдите $x$.",
         "answer": "0,9940", "exams_id": 180, "topics_id": 526, "page_numbers": [1]},
        {"task_num": 2, "source_label": "C1", "condition": r"А) Решите $\begin{cases}x=1\\y=2\end{cases}$. Б) Найдите $x$.",
         "answer": r"$\sqrt{2}$", "exams_id": 180, "topics_id": "Новая осмысленная тема", "page_numbers": [1]},
    ]


def snapshot(ctx):
    return ({path.name: path.read_bytes() for path in ctx.output_dir.iterdir()}, ctx.archive_path.read_bytes())


def test_two_variants_with_same_task_and_image_names_have_isolated_excel_zip_and_metadata(tmp_path, source):
    pdf, positions = source
    first, second = context(tmp_path, 1), context(tmp_path, 2)
    original = tasks(1)
    result1 = export_variant(first, original, {}, pdf_path=pdf, positions=positions)
    result2 = export_variant(second, tasks(2), {}, pdf_path=pdf, positions=positions)
    assert "image_name" not in original[0]
    assert result1["image_count"] == result2["image_count"] == 2
    assert [path.name for path in result1["image_files"]] == [path.name for path in result2["image_files"]]
    for ctx in (first, second):
        workbook = load_workbook(ctx.output_dir / "tasks.xlsx")
        try:
            assert workbook["Tasks"].max_column == 7
            assert workbook["Tasks"].max_row == 3
            assert len(workbook["Tasks"]["C2"].value.splitlines()) == 2
            assert f"Вариант ${ctx.number}$" in workbook["Tasks"]["B2"].value
            assert workbook["Tasks"]["E2"].value == "0,9940"
            assert workbook["Tasks"]["E2"].data_type == "s"
            assert workbook["Tasks"]["D2"].value is None
            assert r"x=1\\y=2" in workbook["Tasks"]["B3"].value
            assert "<br>Б)" in workbook["Tasks"]["B3"].value
            assert tuple(cell.value for cell in workbook["about"][1]) == ABOUT_FIELDS
            about = dict(zip(ABOUT_FIELDS, [cell.value for cell in workbook["about"][2]]))
            assert about["source_url"] == ctx.source_url
            assert about["year"] == 2013
            assert about["date"] is None
        finally:
            workbook.close()
        with ZipFile(ctx.archive_path) as archive:
            assert len(archive.namelist()) == 3
            assert set(archive.namelist()) == {"tasks.xlsx", *[path.name for path in result1["image_files"]]}
            assert archive.testzip() is None


@pytest.mark.parametrize("failure_at", ["excel", "publish_archive"])
def test_failed_new_export_preserves_previous_output_and_zip(tmp_path, source, monkeypatch, failure_at):
    pdf, positions = source
    ctx = context(tmp_path, 1)
    export_variant(ctx, tasks(1), {}, pdf_path=pdf, positions=positions)
    previous = snapshot(ctx)
    if failure_at == "excel":
        def fail(*args, **kwargs):
            raise OSError("disk write failed")
        monkeypatch.setattr(variant_export, "write_excel", fail)
    else:
        original_replace = Path.replace
        def fail_pending_archive(self, target):
            if self.name == "result.zip" and Path(target) == ctx.archive_path:
                raise OSError("archive replace failed")
            return original_replace(self, target)
        monkeypatch.setattr(Path, "replace", fail_pending_archive)
    with pytest.raises(OSError):
        export_variant(ctx, tasks(2), {}, pdf_path=pdf, positions=positions)
    assert snapshot(ctx) == previous


def test_zip_explicit_file_list_excludes_debug_and_rejects_neighbor_variant(tmp_path):
    own = tmp_path / "one"
    own.mkdir()
    other = tmp_path / "two"
    other.mkdir()
    excel = own / "tasks.xlsx"
    excel.write_bytes(b"book")
    image = own / "same.png"
    image.write_bytes(b"own picture")
    neighbor = other / "same.png"
    neighbor.write_bytes(b"other picture")
    (own / "debug.json").write_text("{}")
    archive = tmp_path / "result.zip"
    assert create_zip(excel, own, archive, image_files=[image]) == 2
    previous = archive.read_bytes()
    with ZipFile(archive) as result:
        assert result.namelist() == ["tasks.xlsx", "same.png"]
        assert result.read("same.png") == b"own picture"
    with pytest.raises(ValueError, match="вне папки"):
        create_zip(excel, own, archive, image_files=[neighbor])
    assert archive.read_bytes() == previous


@pytest.mark.parametrize(("value", "expected"), [
    ("0,9940", "0,9940"),
    ("3/7", r"$\frac{3}{7}$"),
    ("√43", r"$\sqrt{43}$"),
    ("(-∞; 2]", r"$(-\infty ; 2]$"),
    (r"$\begin{cases}x=1\\y=2\end{cases}$", r"$\begin{cases}x=1\\y=2\end{cases}$"),
    (r"А) $\sqrt{3}$. Б) $2$.", r"А) $\sqrt{3}$.<br>Б) $2$."),
])
def test_published_answers_keep_strings_and_latex_without_solving(value, expected):
    assert normalize_answer(value) == expected


def test_missing_answer_exports_review_result_and_original_formula_is_unchanged(tmp_path, source):
    pdf, positions = source
    given = tasks(1)
    given[1]["answer"] = ""
    result = export_variant(context(tmp_path, 1), given, {}, pdf_path=pdf, positions=positions)
    assert any("C1" in warning and "отсутствует" in warning for warning in result["warnings"])


def test_542_keeps_four_image_names_and_original_bytes_except_bitmap_with_soft_mask(tmp_path):
    pdf = Path(__file__).resolve().parents[1] / "input" / "trvar542.pdf"
    if not pdf.exists():
        pytest.skip("локальный проверенный PDF 542 не установлен")
    from recognizer import get_task_positions
    positions = {}
    with pymupdf.open(pdf) as document:
        for number, page in enumerate(document, 1):
            positions[number] = [{**item, "source_label": str(item["task_num"])} for item in get_task_positions(page)]
    given = [{"task_num": number, "source_label": str(number), "page_numbers": []} for number in range(1, 20)]
    actual, warnings, manifest = extract_variant_images(pdf, given, tmp_path / "images", positions=positions)
    expected = {"1": ["image_82.jpeg"], "3": ["image_83.jpeg"], "8": ["image_198.png"], "11": ["image_200.png"]}
    assert {key: value for key, value in actual.items() if value} == expected
    assert not warnings
    assert len(manifest) == 4
    with pymupdf.open(pdf) as document:
        for names in expected.values():
            name = names[0]
            xref = int(name.split("_")[1].split(".")[0])
            if xref != 198:
                assert (tmp_path / "images" / name).read_bytes() == document.extract_image(xref)["image"]
            else:
                # Старый файл198 сохранял сырые RGB и игнорировал soft mask.
                # Новое извлечение корректно применяет её по общему правилу;
                # существующие output/ZIP пользователя этот тест не трогает.
                rendered = pymupdf.Pixmap(str(tmp_path / "images" / name))
                assert rendered.pixel(0, 0) == (255, 255, 255)
                entry = next(item for item in manifest if item["image_name"] == name)
                assert entry["soft_mask"] > 0 and entry["rendered_placement"] is True


def test_image_before_first_label_in_next_column_belongs_to_continued_task():
    anchors = [{"source_label": "B1", "x": 30, "y": 40},
               {"source_label": "B2", "x": 30, "y": 220},
               {"source_label": "C1", "x": 430, "y": 190}]
    figure = pymupdf.Rect(510, 50, 600, 120)
    given = [{"source_label": label, "page_numbers": [1]} for label in ("B1", "B2", "C1")]
    assert variant_export._owner(figure, anchors, given, 1, 800) == "B2"


def test_image_continuing_task_on_next_page_does_not_need_repeated_label():
    given = [{"source_label": "C1", "page_numbers": [1, 2]},
             {"source_label": "C2", "page_numbers": [2]}]
    anchors = [{"source_label": "C2", "x": 30, "y": 250}]
    assert variant_export._owner(pymupdf.Rect(100, 30, 190, 120), anchors, given, 2, 400) == "C1"


def test_vector_detection_excludes_table_grid_and_tall_radical_but_keeps_geometry():
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=600)
        for x in range(30, 151, 40):
            page.draw_line((x, 30), (x, 110))
        for y in range(30, 111, 40):
            page.draw_line((30, y), (150, y))
        # Большой корень, в том числе его короткая наклонная засечка.
        page.draw_polyline([(210, 120), (215, 125), (220, 65), (340, 65)])
        page.draw_line((230, 100), (330, 100))
        # Настоящий векторный треугольник.
        page.draw_polyline([(100, 250), (170, 350), (240, 250), (100, 250)])
        rects = variant_export._vector_rects(page)
        assert len(rects) == 1
        assert rects[0].contains(pymupdf.Point(170, 300))


def test_table_preserves_cells_and_does_not_split_modulus_in_math():
    table = "Тарифы\n| Вид | Условие |\n|---|---|\n| A | $|x|<2$ |\n| Б | 0,96 |"
    rendered = variant_export.format_condition_tables(table)
    assert rendered.startswith("Тарифы\n<table><tr><th>Вид</th><th>Условие</th>")
    assert "<td>$|x|&lt;2$</td>" in rendered
    assert "<td>0,96</td>" in rendered
    assert rendered.count("<tr>") == 3


def asymmetric_picture(*, transparent=False):
    """Разные углы позволяют проверить поворот без решения задачи по картинке."""
    with pymupdf.open() as document:
        page = document.new_page(width=40, height=30)
        page.draw_rect((0, 0, 20, 15), color=None, fill=(1, 0, 0))
        page.draw_rect((30, 0, 40, 10), color=None, fill=(0, 0, 1))
        return page.get_pixmap(alpha=transparent).tobytes("png")


@pytest.mark.parametrize("page_rotation", [0, 90, 180, 270])
def test_placed_bitmap_matches_pdf_rotation_and_soft_mask_without_nearby_text(tmp_path, page_rotation):
    path = tmp_path / "masked.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=240, height=240)
        page.insert_image((80, 80, 140, 160), stream=asymmetric_picture(transparent=True),
                          rotate=90, keep_proportion=False)
        page.insert_text((10, 20), "NEARBY TEXT MUST NOT BE EXPORTED")
        page.set_rotation(page_rotation)
        item = page.get_images(full=True)[0]
        assert item[1] > 0  # Реальная soft mask, а не заранее белый фон.
        rectangle, _ = page.get_image_rects(item[0], transform=True)[0]
        rectangle *= page.rotation_matrix
        expected = page.get_pixmap(matrix=pymupdf.Matrix(0.5, 0.5), clip=rectangle, alpha=False)
        anchors = {1: [{"source_label": "B1", "x": rectangle.x0 - 2, "y": rectangle.y0 - 10}]}
        document.save(path)
    actual, warnings, manifest = extract_variant_images(
        path, [{"source_label": "B1", "page_numbers": [1]}], tmp_path / "images", positions=anchors,
    )
    rendered = pymupdf.Pixmap(str(tmp_path / "images" / actual["B1"][0]))
    assert (rendered.width, rendered.height) == (expected.width, expected.height)
    assert rendered.samples == expected.samples
    assert rendered.alpha == 0
    assert not warnings
    assert len(manifest) == 1
    assert manifest[0]["soft_mask"] > 0 and manifest[0]["rendered_placement"] is True


def test_same_xref_with_two_orientations_gets_two_correct_task_images_and_keeps_original_bytes(tmp_path):
    path = tmp_path / "reused.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=350)
        xref = page.insert_image((50, 60, 130, 120), stream=asymmetric_picture())
        page.insert_image((50, 180, 110, 260), xref=xref, rotate=90, keep_proportion=False)
        original = document.extract_image(xref)["image"]
        expected = page.get_pixmap(matrix=pymupdf.Matrix(0.5, 0.5), clip=pymupdf.Rect(50, 180, 110, 260), alpha=False)
        document.save(path)
    anchors = {1: [{"source_label": "B1", "x": 30, "y": 40},
                   {"source_label": "B2", "x": 30, "y": 160}]}
    actual, warnings, manifest = extract_variant_images(
        path, [{"source_label": label, "page_numbers": [1]} for label in ("B1", "B2")],
        tmp_path / "images", positions=anchors,
    )
    assert len(actual["B1"]) == len(actual["B2"]) == 1
    assert actual["B1"] != actual["B2"]
    assert (tmp_path / "images" / actual["B1"][0]).read_bytes() == original
    rotated = pymupdf.Pixmap(str(tmp_path / "images" / actual["B2"][0]))
    assert rotated.samples == expected.samples
    assert len(manifest) == 2 and not warnings


def test_geometric_b1_inside_later_task_does_not_create_false_column_or_steal_images(tmp_path):
    path = tmp_path / "geometric-label.pdf"
    with pymupdf.open() as document:
        document.new_page(width=842, height=595)
        page = document.new_page(width=842, height=595)
        for rectangle, colour in [((650, 60, 750, 120), (1, 0, 0)),
                                   ((650, 210, 750, 270), (0, 1, 0)),
                                   ((730, 460, 820, 530), (0, 0, 1))]:
            page.insert_image(rectangle, stream=picture(colour))
        document.save(path)
    positions = {1: [{"source_label": "B1", "x": 30, "y": 40}],
                 2: [{"source_label": "B3", "x": 30, "y": 40},
                     {"source_label": "B8", "x": 438, "y": 40},
                     {"source_label": "B9", "x": 438, "y": 180},
                     {"source_label": "B1", "x": 525, "y": 204},
                     {"source_label": "B11", "x": 438, "y": 440}]}
    given = [{"source_label": label, "page_numbers": [1] if label == "B1" else [2]}
             for label in ("B1", "B3", "B8", "B9", "B11")]
    actual, warnings, manifest = extract_variant_images(path, given, tmp_path / "images", positions=positions)
    assert actual["B1"] == []
    assert {label: len(names) for label, names in actual.items() if names} == {"B8": 1, "B9": 1, "B11": 1}
    assert [item["source_label"] for item in manifest] == ["B8", "B9", "B11"]
    assert not warnings
