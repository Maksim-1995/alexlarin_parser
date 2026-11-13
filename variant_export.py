"""Изолированная сборка и проверка Excel, рисунков и ZIP одного варианта."""

import json
import math
import re
import shutil
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pymupdf
from openpyxl import load_workbook

from export_metadata import ABOUT_FIELDS, prepare_about_metadata
from exporter import create_zip, write_excel
from latex_format import MATH_RE, normalize_condition
from task_html import add_subtask_breaks


TASK_FIELDS = (
    "task_num", "condition", "image_name", "solution", "answer", "exams_id", "topics_id",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jpx", ".jp2", ".bmp", ".tiff", ".gif", ".webp"}


def format_condition_tables(text):
    """Передать распознанную таблицу в HTML без изменения её ячеек.

    Разделителями являются только | вне MathJax; модуль внутри $...$ не
    становится новой колонкой. Неоднозначная разметка остаётся исходным текстом.
    """
    def cells(line):
        result = [""]
        for index, part in enumerate(MATH_RE.split(line.strip().strip("|"))):
            if index % 2:
                result[-1] += part
            else:
                bits = part.split("|")
                result[-1] += bits[0]
                result.extend(bits[1:])
        return [cell.strip() for cell in result]

    lines, result, index = text.splitlines(), [], 0
    separator = re.compile(r"\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*")
    while index < len(lines):
        if index + 1 < len(lines) and "|" in lines[index] and separator.fullmatch(lines[index + 1]):
            rows = [cells(lines[index])]
            end = index + 2
            while end < len(lines) and lines[end].strip().startswith("|"):
                row = cells(lines[end])
                if len(row) != len(rows[0]):
                    break
                rows.append(row)
                end += 1
            if len(rows) > 1:
                table = "<table>" + "".join(
                    "<tr>" + "".join(f"<{tag}>{escape(value)}</{tag}>" for value in row) + "</tr>"
                    for row, tag in [(rows[0], "th"), *((r, "td") for r in rows[1:])]
                ) + "</table>"
                result.append(table)
                index = end
                continue
        result.append(lines[index])
        index += 1
    return "\n".join(result)


def normalize_answer(answer):
    """Оформить опубликованную запись, не вычисляя и не упрощая ответ.

    Обычные числа остаются строками, включая десятичную запятую. Уже
    размеченный LaTeX не меняется. Для простой линейной записи разрешены
    только однозначные преобразования знаков, дроби и корня из числа.
    """
    if not isinstance(answer, str):
        raise ValueError("Ответ должен быть строкой")
    answer = answer.strip()
    if not answer:
        return ""
    parts = MATH_RE.split(answer)
    for index in range(0, len(parts), 2):
        text = parts[index]
        if re.search(r"(?<!\\)\$", text):
            raise ValueError("Непарный знак $ в ответе")
        parts[index] = re.sub(r"[ \t\u00a0]+", " ", text)
    answer = "".join(parts)
    if len(parts) > 1:
        return add_subtask_breaks(answer)

    # Подпункты ответа также принадлежат одной основной задаче.
    subparts = re.split(r"(?<!\S)([АБВГABCDабвгabcd]\))", answer)
    if len(subparts) > 1:
        result = normalize_answer(subparts[0])
        for index in range(1, len(subparts), 2):
            result += " " + subparts[index] + " " + normalize_answer(subparts[index + 1])
        return add_subtask_breaks(result.strip())

    if re.fullmatch(r"[+−-]?\d+(?:[.,]\d+)?", answer):
        return answer
    fraction = re.fullmatch(r"([+−-]?\d+(?:[.,]\d+)?)\s*/\s*([+−-]?\d+(?:[.,]\d+)?)", answer)
    if fraction:
        return rf"$\frac{{{fraction[1]}}}{{{fraction[2]}}}$"

    # Русский текст остаётся текстом. Команда без математических границ
    # в таком тексте требует проверки, а не угадывания границ выражения.
    if re.search(r"[А-Яа-яЁё]", answer) and not re.fullmatch(r"\\text\{.*\}", answer):
        if re.search(r"\\[A-Za-z]+", answer):
            raise ValueError("Команда LaTeX вне формулы в текстовом ответе")
        return answer

    rendered = re.sub(r"√\s*(\d+(?:[.,]\d+)?)", r"\\sqrt{\1}", answer)
    if "√" in rendered:
        raise ValueError("Неясная область корня в опубликованном ответе")
    symbols = {
        "π": r"\pi ", "∞": r"\infty ", "≤": r"\le ", "≥": r"\ge ",
        "≠": r"\ne ", "∪": r"\cup ", "∩": r"\cap ", "∈": r"\in ",
        "±": r"\pm ", "−": "-", "×": r"\times ", "·": r"\cdot ",
    }
    for symbol, command in symbols.items():
        rendered = rendered.replace(symbol, command)
    if re.search(r"\\[A-Za-z]+|[\^_=<>/]|[\[\(].*;.*[\]\)]|\d\s*[+*-]\s*\d", rendered):
        return "$" + rendered.strip() + "$"
    return answer


def _column_starts(anchors, page_width):
    """Границы колонок выводятся из положений меток, а не из номера варианта."""
    groups = []
    tolerance = max(35, page_width * 0.09)
    for x in sorted(anchor["x"] for anchor in anchors):
        if not groups or x - groups[-1] > tolerance:
            groups.append(x)
    return [max(0, x - 16) for x in groups]


def _owner(rect, anchors, tasks, page_number, page_width):
    center_x, center_y = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
    starts = _column_starts(anchors, page_width)
    if not starts:
        # На странице может быть только продолжение условия.
        continued = [task for task in tasks if page_number in task.get("page_numbers", [])
                     and min(task.get("page_numbers") or [page_number]) < page_number]
        return continued[0]["source_label"] if len(continued) == 1 else None
    column = next((i for i in reversed(range(len(starts))) if center_x >= starts[i]), None)
    if column is None:
        return None
    left = starts[column]
    right = starts[column + 1] if column + 1 < len(starts) else page_width
    same_column = [item for item in anchors if left <= item["x"] < right]
    above = [item for item in same_column if item["y"] <= center_y + 3]
    if above:
        return max(above, key=lambda item: item["y"])["source_label"]
    # Продолжение задачи может начинаться с рисунка в соседней колонке
    # перед первой новой меткой. Это всё ещё та же задача.
    if column > 0:
        previous = [item for item in anchors if item["x"] < left]
        if previous:
            previous_starts = starts[:column]
            last_start = previous_starts[-1]
            return max((item for item in previous if item["x"] >= last_start),
                       key=lambda item: item["y"])["source_label"]
    continued = [task for task in tasks if page_number in task.get("page_numbers", [])
                 and min(task.get("page_numbers") or [page_number]) < page_number]
    if len(continued) == 1 and column == 0:
        return continued[0]["source_label"]
    return None


def _page_positions(page, page_number, positions):
    if positions is None:
        from variant_sources import task_positions
        return task_positions(page)
    if isinstance(positions, dict):
        return positions.get(page_number, [])
    return positions[page_number - 1]


def _vector_rects(page):
    """Найти связанные векторные контуры, отсеяв текст и простые рамки.

    Заливные контуры обычно являются водяными знаками и глифами; их не
    превращаем в рисунки. Короткие линии дробей и корней отсеиваются по
    размеру. Растровые рисунки обрабатываются независимо.
    """
    drawings = [item for item in page.get_drawings()
                if item.get("type") in {"s", "fs"}
                and not (len(item["items"]) == 1 and item["items"][0][0] == "re")]
    if not drawings:
        return []
    groups = page.cluster_drawings(drawings=drawings, x_tolerance=4, y_tolerance=4)
    result = []
    for rect in groups:
        if rect.width < 35 or rect.height < 45:
            continue
        members = [item for item in drawings if (item["rect"] + (-1, -1, 1, 1)).intersects(rect)]
        segments = sum(len(item["items"]) for item in members)
        # Табличные сетки, дробные черты, скобки и даже высокий корень
        # образуют связные контуры. У них нет протяжённых диагоналей или
        # широких кривых, характерных для геометрии и графиков.
        graphic_stroke = False
        for member in members:
            for item in member["items"]:
                if item[0] == "l":
                    graphic_stroke |= abs(item[1].x - item[2].x) >= 15 and abs(item[1].y - item[2].y) >= 15
                elif item[0] == "c":
                    xs, ys = [point.x for point in item[1:]], [point.y for point in item[1:]]
                    graphic_stroke |= max(xs) - min(xs) >= 20 and max(ys) - min(ys) >= 20
        if segments >= 3 and graphic_stroke:
            result.append(rect)
    return result


def _image_placement_identity(xref, soft_mask, width, height, matrix):
    """Отличить ориентации одного XObject без зависимости от его положения.

    У обычного рисунка сохраняются исходные байты. Поворот, отражение,
    существенное растяжение и soft mask требуют отдельного рендера. Масштаб
    и перенос сами по себе не создают ещё один файл одинакового рисунка.
    """
    a, b, c, d, _, _ = matrix
    horizontal, vertical = math.hypot(a, b), math.hypot(c, d)
    if horizontal <= 0 or vertical <= 0 or abs(a * d - b * c) < 1e-9:
        raise ValueError(f"Вырожденная матрица рисунка xref={xref}")
    ordinary = (not soft_mask and a > 0 and d > 0 and abs(b) < 1e-6 and abs(c) < 1e-6
                and math.isclose(horizontal / width, vertical / height, rel_tol=0.02))
    if ordinary:
        return xref, "original"
    return xref, soft_mask, tuple(round(value / horizontal, 7) for value in (a, b, c, d))


def _render_image_placement(document, xref, soft_mask, width, height, matrix):
    """Отрисовать только bitmap с его маской и матрицей, без текста PDF.

    Отдельная страница содержит ровно один XObject. Его PDF-матрица
    воспроизводит итоговую геометрию после page.rotation_matrix, включая
    отражения и наклон. Это не снимок области условия или водяного знака.
    """
    pixmap = pymupdf.Pixmap(document, xref)
    if pixmap.colorspace is not None and pixmap.colorspace.n != 3:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    if soft_mask:
        if pixmap.alpha:
            pixmap = pymupdf.Pixmap(pixmap, 0)
        pixmap = pymupdf.Pixmap(pixmap, pymupdf.Pixmap(document, soft_mask))
    png = pixmap.tobytes("png")
    bounds = pymupdf.Rect(0, 0, 1, 1) * matrix
    a, b, c, d, e, f = matrix
    e, f = e - bounds.x0, f - bounds.y0
    # get_image_rects описывает изображение сверху вниз. В content stream
    # PDF ось y направлена вверх, поэтому обе системы координат согласуются
    # явно, без угадывания угла по ширине и высоте картинки.
    pdf_matrix = (a, -b, -c, d, e + c, bounds.height - f - d)
    density = max(width / math.hypot(a, b), height / math.hypot(c, d))
    with pymupdf.open() as isolated:
        page = isolated.new_page(width=bounds.width, height=bounds.height)
        page.insert_image(page.rect, stream=png, keep_proportion=False)
        resource_name = page.get_images(full=True)[0][7]
        # PDF number syntax, в отличие от Python, не допускает экспоненту e-10.
        coefficients = " ".join(f"{value:.10f}" for value in pdf_matrix)
        isolated.update_stream(page.get_contents()[0],
                               f"q\n{coefficients} cm\n/{resource_name} Do\nQ\n".encode("ascii"))
        rendered = page.get_pixmap(matrix=pymupdf.Matrix(density, density),
                                   colorspace=pymupdf.csRGB, alpha=False)
        return {"ext": "png", "image": rendered.tobytes("png")}


def extract_variant_images(pdf_path, tasks, images_dir, *, positions=None):
    """Сохранить рисунки с явной привязкой к source_label данного варианта."""
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    by_label = {task["source_label"]: [] for task in tasks}
    task_pages = {task["source_label"]: set(task.get("page_numbers") or ()) for task in tasks}
    warnings, manifest, saved, placement_counts = [], [], {}, {}
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, 1):
            anchors = _page_positions(page, page_number, positions)
            # Геометрическое обозначение B1 внутри B9 может оказаться первым
            # словом PDF text block. OCR уже установил реальные страницы B1:
            # такая внутренняя надпись не должна создавать новую колонку.
            # У настоящего продолжения текущая страница есть в page_numbers.
            anchors = [anchor for anchor in anchors if anchor["source_label"] in by_label
                       and (not task_pages[anchor["source_label"]]
                            or page_number in task_pages[anchor["source_label"]])]
            embedded_rects = []
            # Один xref может иметь несколько ресурсных имён на странице.
            # get_image_rects уже перечисляет все его реальные размещения.
            for item in {item[0]: item for item in page.get_images(full=True)}.values():
                xref, soft_mask, width, height = item[:4]
                for source_rect, source_matrix in page.get_image_rects(xref, transform=True):
                    rect = source_rect * page.rotation_matrix
                    if rect.width < 12 or rect.height < 12:
                        continue
                    if rect.get_area() > page.rect.get_area() * 0.75:
                        warnings.append(f"Страница {page_number}: скан страницы требует отдельной проверки рисунков")
                        continue
                    if rect.y0 >= page.rect.height - 18:
                        continue
                    label = _owner(rect, anchors, tasks, page_number, page.rect.width)
                    if label not in by_label:
                        continue
                    matrix = source_matrix * page.rotation_matrix
                    identity = _image_placement_identity(xref, soft_mask, width, height, matrix)
                    if identity not in saved:
                        image = (document.extract_image(xref) if identity[1] == "original" else
                                 _render_image_placement(document, xref, soft_mask, width, height, matrix))
                        placement_counts[xref] = placement_counts.get(xref, 0) + 1
                        suffix = "" if placement_counts[xref] == 1 else f"_placement{placement_counts[xref]}"
                        name = f"image_{xref}{suffix}.{image['ext']}"
                        (images_dir / name).write_bytes(image["image"])
                        saved[identity] = name
                    name = saved[identity]
                    if name not in by_label[label]:
                        by_label[label].append(name)
                    embedded_rects.append(rect)
                    manifest.append({"source_label": label, "page_number": page_number,
                                     "image_name": name, "rect": list(rect), "kind": "embedded",
                                     "placement_matrix": list(matrix), "soft_mask": soft_mask,
                                     "rendered_placement": identity[1] != "original"})

            for index, source_rect in enumerate(_vector_rects(page), 1):
                rect = source_rect * page.rotation_matrix
                if any((existing + (-3, -3, 3, 3)).contains(rect) for existing in embedded_rects):
                    continue
                label = _owner(rect, anchors, tasks, page_number, page.rect.width)
                if label not in by_label or rect.y0 >= page.rect.height - 18:
                    continue
                # Поля дают место обозначениям у вершин, но не всему условию.
                clip = (rect + (-12, -12, 12, 12)) & page.rect
                name = f"figure_p{page_number:02d}_{index:02d}.png"
                page.get_pixmap(dpi=200, clip=clip, alpha=False).save(images_dir / name)
                by_label[label].append(name)
                manifest.append({"source_label": label, "page_number": page_number,
                                 "image_name": name, "rect": list(clip), "kind": "vector"})
                warnings.append(f"{label}, страница {page_number}: проверьте границы векторного рисунка {name}")
    return by_label, warnings, manifest


def _validate_result(tasks, metadata, directory, archive_path):
    """Проверить контракт Excel, ссылки и точный состав выдачи."""
    directory = Path(directory)
    numbers = [task["task_num"] for task in tasks]
    if numbers != list(range(1, len(tasks) + 1)) or not numbers:
        raise ValueError("Нужны все последовательные task_num одного варианта")
    labels = [task["source_label"] for task in tasks]
    if len(labels) != len(set(labels)):
        raise ValueError("Повторная исходная метка задачи")
    referenced = set()
    for task in tasks:
        if not isinstance(task.get("condition"), str) or not task["condition"].strip():
            raise ValueError(f"Пустое условие задачи {task['task_num']}")
        if not isinstance(task.get("answer"), str):
            raise ValueError("Ответы должны быть строками")
        if type(task.get("exams_id")) is not int or not isinstance(task.get("topics_id"), (str, int)):
            raise ValueError("Нет проверенной классификации задачи")
        for name in task.get("image_name", "").splitlines():
            if Path(name).name != name or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Некорректное имя рисунка: {name}")
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
            referenced.add(name)
    expected = {"tasks.xlsx", *referenced}
    if {path.name for path in directory.iterdir()} != expected:
        raise ValueError("В выдаче есть посторонние файлы или отсутствуют рисунки")
    workbook = load_workbook(directory / "tasks.xlsx")
    try:
        if workbook.sheetnames != ["Tasks", "about"]:
            raise ValueError("Неверный набор листов Excel")
        sheet = workbook["Tasks"]
        if tuple(cell.value for cell in sheet[1]) != TASK_FIELDS or sheet.max_row != len(tasks) + 1:
            raise ValueError("Неверный формат Tasks")
        for row, task in zip(sheet.iter_rows(min_row=2), tasks):
            for index, field in enumerate(TASK_FIELDS):
                expected_value = None if field == "solution" else task.get(field) or None
                if row[index].value != expected_value:
                    raise ValueError(f"Excel изменил значение {field} задачи {task['task_num']}")
            if row[4].value is not None and row[4].data_type != "s":
                raise ValueError("Ответ в Excel должен сохраняться строкой")
        about = workbook["about"]
        if tuple(cell.value for cell in about[1]) != ABOUT_FIELDS or about.max_row != 2:
            raise ValueError("Неверный формат about")
        if about.cell(2, ABOUT_FIELDS.index("source_url") + 1).value != metadata["source_url"]:
            raise ValueError("Excel относится к другому источнику")
    finally:
        workbook.close()
    with ZipFile(archive_path) as archive:
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError("ZIP содержит файлы другого варианта или лишние файлы")
        if archive.testzip() is not None:
            raise ValueError("ZIP повреждён")
        for name in expected:
            if archive.read(name) != (directory / name).read_bytes():
                raise ValueError(f"Файл {name} не совпадает с ZIP")


def _publish(directory, pending_archive, context):
    """Заменить оба результата с откатом при ошибке файловой операции."""
    output, archive = Path(context.output_dir), Path(context.archive_path)
    backup = Path(context.debug_dir) / "previous_results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    previous_output, previous_archive = backup / "output", backup / "result.zip"
    moved_output = moved_archive = new_output = new_archive = False
    try:
        if output.exists() or archive.exists():
            backup.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.rename(previous_output)
            moved_output = True
        if archive.exists():
            archive.rename(previous_archive)
            moved_archive = True
        Path(directory).rename(output)
        new_output = True
        Path(pending_archive).replace(archive)
        new_archive = True
    except BaseException:
        if new_archive:
            archive.unlink(missing_ok=True)
        if new_output:
            shutil.rmtree(output)
        if moved_output:
            previous_output.rename(output)
        if moved_archive:
            previous_archive.rename(archive)
        raise


def export_variant(context, tasks, metadata, *, pdf_path, positions=None):
    """Собрать проверяемую выдачу, не меняя задачи и прежний результат заранее.

    Все дополнительные сведения остаются в debug/, а папка output/ и ZIP
    содержат только семь полей Tasks, согласованный about и рисунки.
    Возвращаемые предупреждения предназначены для статуса NEEDS_REVIEW.
    """
    if metadata.get("source_url", context.source_url) != context.source_url:
        raise ValueError("Метаданные относятся к другому варианту")
    if metadata.get("year", context.year) != context.year:
        raise ValueError("Год метаданных не соответствует варианту")
    about = prepare_about_metadata({
        **metadata, "year": context.year, "source_url": context.source_url,
        "title": metadata.get("title") or f"Тренировочный вариант № {context.number}",
        "code": context.code,
    })
    prepared = [dict(task) for task in tasks]
    warnings = []
    output, archive = Path(context.output_dir), Path(context.archive_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    Path(context.debug_dir).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{context.variant_key}-") as temporary:
        stage_root = Path(temporary)
        directory = stage_root / "result"
        directory.mkdir()
        image_map, image_warnings, manifest = extract_variant_images(pdf_path, prepared, directory, positions=positions)
        warnings.extend(image_warnings)
        for task in prepared:
            task["image_name"] = "\n".join(image_map[task["source_label"]])
            if not task["image_name"] and re.search(
                r"на\s+рисунк[еах]|на\s+чертеже|(?:на|по)\s+график[еу]|изображ[её]н.{0,30}график",
                task["condition"], re.IGNORECASE,
            ):
                warnings.append(f"{task['source_label']}: условие ссылается на рисунок, извлечение требует проверки по PDF")
            condition, annotations = re.subn(
                r"\[ИЗОБРАЖЕНИЕ\s*:[^\]]*\]", "", task["condition"], flags=re.IGNORECASE,
            )
            if annotations:
                warnings.append(f"{task['source_label']}: удалено добавленное OCR описание рисунка; сверьте условие с PDF")
                task["condition"] = condition.strip()
            try:
                task["condition"] = format_condition_tables(task["condition"])
                task["condition"] = add_subtask_breaks(normalize_condition(task["condition"]))
            except ValueError as error:
                warnings.append(f"{task['source_label']}: условие требует проверки оформления: {error}")
            try:
                task["answer"] = normalize_answer(task.get("answer", ""))
            except ValueError as error:
                warnings.append(f"{task['source_label']}: ответ требует проверки оформления: {error}")
            if not task.get("answer", "").strip():
                warnings.append(f"{task['source_label']}: опубликованный ответ отсутствует")
        write_excel(prepared, about, directory / "tasks.xlsx")
        pending_archive = stage_root / "result.zip"
        names = sorted({name for task in prepared for name in task["image_name"].splitlines()})
        create_zip(directory / "tasks.xlsx", directory, pending_archive,
                   image_files=[directory / name for name in names])
        _validate_result(prepared, about, directory, pending_archive)
        _publish(directory, pending_archive, context)

    # Проверочные сведения не входят в Excel/ZIP. Пишем после успешной
    # выдачи через временный файл; неудача отчёта не меняет сам результат.
    report = {"variant_key": context.variant_key, "source_url": context.source_url,
              "images": manifest, "warnings": warnings,
              "tasks": [{"task_num": task["task_num"], "source_label": task["source_label"],
                         "page_numbers": task.get("page_numbers", []), "image_name": task["image_name"]}
                        for task in prepared]}
    target = Path(context.debug_dir) / "export_verification.json"
    pending = target.with_suffix(".json.tmp")
    try:
        pending.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)
    return {"excel_path": output / "tasks.xlsx", "archive_path": archive,
            "image_files": [output / name for name in names], "image_count": len(names),
            "warnings": warnings, "verification_path": target}
