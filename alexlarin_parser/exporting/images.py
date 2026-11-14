"""Рисунки одного варианта: принадлежность, геометрия PDF и bitmap-маски."""

import math
from pathlib import Path

import pymupdf


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
        from ..sources.parsing import task_positions
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
