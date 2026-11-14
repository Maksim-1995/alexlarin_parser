"""Сборка, проверка и публикация изолированной выдачи одного варианта."""

import json
import re
import shutil
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from openpyxl import load_workbook

from ..formatting.latex import MATH_RE, normalize_condition
from ..formatting.html import add_subtask_breaks
from .metadata import ABOUT_FIELDS, prepare_about_metadata
from .archive import create_zip
from .workbook import write_excel
from .images import extract_variant_images


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
