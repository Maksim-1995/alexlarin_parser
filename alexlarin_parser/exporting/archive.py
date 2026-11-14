"""Атомарная упаковка явного набора итоговых файлов варианта."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


def create_zip(excel_path, images_dir, archive_path, *, image_files=None):
    """Создать ZIP-архив и проверить целостность записанных файлов.

    Книга сохраняется в архиве как ``tasks.xlsx``. Рисунки добавляются
    рядом с ней, без вложенной папки. ``images_dir`` может совпадать
    с папкой Excel: книга повторно не добавляется. Сам архив также
    исключается из списка исходных файлов. Вложенные каталоги
    не обходятся. Существующий архив перезаписывается.

    Каталог изображений должен содержать только итоговые рисунки
    и, при совпадении с папкой результата, книгу Excel.

    Args:
        excel_path: Путь к книге Excel, добавляемой в архив.
        images_dir: Каталог изображений; теперь это папка результата.
        archive_path: Путь к создаваемому ZIP-архиву.

    Returns:
        int: Количество файлов в архиве, включая книгу Excel.

    Raises:
        OSError: Не удалось прочитать исходные файлы или записать архив.
        ValueError: Проверка целостности обнаружила повреждённый файл.
    """
    excel_path = Path(excel_path)
    images_dir = Path(images_dir)
    archive_path = Path(archive_path)
    excluded_paths = {excel_path.resolve(), archive_path.resolve()}
    # Старый публичный API оставлен для проверенной выгрузки 542.
    # Новый пакетный путь всегда передаёт явный список итоговых рисунков.
    candidates = images_dir.iterdir() if image_files is None else image_files
    images = []
    names = {"tasks.xlsx"}
    for candidate in candidates:
        image_path = Path(candidate)
        if not image_path.is_absolute() and image_path.parent == Path("."):
            image_path = images_dir / image_path
        if image_path.resolve() in excluded_paths:
            continue
        if image_files is None and not image_path.is_file():
            continue
        if image_path.resolve().parent != images_dir.resolve():
            raise ValueError(f"Рисунок вне папки данного варианта: {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if image_path.name in names:
            raise ValueError(f"Повторное имя в ZIP: {image_path.name}")
        names.add(image_path.name)
        images.append(image_path)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with NamedTemporaryFile(
            dir=archive_path.parent, prefix=archive_path.name + ".", suffix=".tmp", delete=False,
        ) as target:
            pending = Path(target.name)
        with ZipFile(pending, mode="w", compression=ZIP_DEFLATED) as archive:
            archive.write(excel_path, arcname="tasks.xlsx")
            for image_path in sorted(images):
                archive.write(image_path, arcname=image_path.name)

        with ZipFile(pending, mode="r") as archive:
            damaged_file = archive.testzip()
            if damaged_file is not None:
                raise ValueError(f"В архиве повреждён файл: {damaged_file}")
            if len(archive.namelist()) != len(names) or set(archive.namelist()) != names:
                raise ValueError("Состав ZIP не совпадает со списком итоговых файлов")
        pending.replace(archive_path)
        return len(names)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
