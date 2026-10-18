"""Экспорт заданий в Excel и ZIP-архив с изображениями."""

from zipfile import ZIP_DEFLATED, ZipFile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill


def write_excel(tasks, metadata, excel_path):
    """Сохранить задания и метаданные в книгу Excel.

    Лист ``Tasks`` содержит поля ``task_num``, ``condition``,
    ``image_name``, ``solution`` и ``answer``. Поле ``solution``
    остаётся пустым. Лист ``about`` содержит пары ключей и значений
    метаданных; для поля ``date`` задаётся формат ``yyyy-mm-dd``.

    Args:
        tasks: Итерируемая коллекция словарей с полями ``task_num``,
            ``condition``, ``image_name`` и ``answer``.
        metadata: Словарь метаданных для листа ``about``.
        excel_path: Путь к сохраняемой книге Excel.

    Raises:
        KeyError: В задании отсутствует обязательное поле.
        OSError: Не удалось записать книгу по указанному пути.
    """
    workbook = Workbook()

    # При создании книги один лист уже существует.
    tasks_sheet = workbook.active
    tasks_sheet.title = "Tasks"

    tasks_sheet.append([
        "task_num",
        "condition",
        "image_name",
        "solution",
        "answer",
    ])

    for task in tasks:
        tasks_sheet.append([
            task["task_num"],
            task["condition"],
            task["image_name"],
            "",  # Решения пока оставляем пустыми.
            task["answer"],
        ])

    about_sheet = workbook.create_sheet("about")
    about_sheet.append(["field", "value"])

    for field, value in metadata.items():
        about_sheet.append([field, value])

        if field == "date":
            date_cell = about_sheet.cell(
                row=about_sheet.max_row,
                column=2,
            )
            date_cell.number_format = "yyyy-mm-dd"

    format_excel(workbook)
    workbook.save(excel_path)


def format_excel(workbook):
    """Оформить листы книги Excel, изменяя её на месте.

    На всех листах закрепляется строка заголовков, включается перенос
    текста и задаётся стиль заголовков. На листах ``Tasks`` и ``about``
    настраивается ширина столбцов; для ``Tasks`` включается автофильтр.

    Args:
        workbook: Книга openpyxl с листами ``Tasks`` и ``about``.

    Raises:
        KeyError: В книге отсутствует лист ``Tasks`` или ``about``.
    """
    # Общие настройки обоих листов.
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"

        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=True,
                )

        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(
                fill_type="solid",
                fgColor="305496",
            )

        sheet.row_dimensions[1].height = 25

    tasks_sheet = workbook["Tasks"]

    column_widths = {
        "A": 12,  # task_num
        "B": 90,  # condition
        "C": 30,  # image_name
        "D": 30,  # solution
        "E": 50,  # answer
    }

    for column, width in column_widths.items():
        tasks_sheet.column_dimensions[column].width = width

    tasks_sheet.auto_filter.ref = tasks_sheet.dimensions

    about_sheet = workbook["about"]
    about_sheet.column_dimensions["A"].width = 30
    about_sheet.column_dimensions["B"].width = 85


def create_zip(excel_path, images_dir, archive_path):
    """Создать ZIP-архив и проверить целостность записанных файлов.

    Книга сохраняется в архиве как ``tasks.xlsx``. Все файлы из
    ``images_dir`` добавляются в каталог ``images`` в порядке имён;
    вложенные каталоги не обходятся. Существующий архив
    перезаписывается.

    Args:
        excel_path: Путь к книге Excel, добавляемой в архив.
        images_dir: Объект pathlib.Path с каталогом изображений.
        archive_path: Путь к создаваемому ZIP-архиву.

    Returns:
        int: Количество файлов в архиве, включая книгу Excel.

    Raises:
        OSError: Не удалось прочитать исходные файлы или записать архив.
        ValueError: Проверка целостности обнаружила повреждённый файл.
    """
    with ZipFile(
        archive_path,
        mode="w",
        compression=ZIP_DEFLATED,
    ) as archive:
        archive.write(excel_path, arcname="tasks.xlsx")

        for image_path in sorted(images_dir.iterdir()):
            if image_path.is_file():
                archive.write(
                    image_path,
                    arcname=f"images/{image_path.name}",
                )

    # Проверяем целостность записанных файлов.
    with ZipFile(archive_path, mode="r") as archive:
        damaged_file = archive.testzip()

        if damaged_file is not None:
            raise ValueError(
                f"В архиве повреждён файл: {damaged_file}"
            )

        return len(archive.namelist())
