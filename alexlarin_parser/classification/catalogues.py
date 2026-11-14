"""Полные деревья exams/topics, допустимые листья и проверка классификации.

CSV проверяются целиком до выделения математической ветки. Этот модуль
не обращается к модели и не читает/записывает AI-кэши.
"""

import csv
from pathlib import Path
import re
import unicodedata


# Набор по умолчанию нужен только совместимому протоколу варианта 542.
# Основной сервис передаёт task_numbers конкретного варианта явно.
TASK_NUMBERS = set(range(1, 20))


class Catalog(list):
    """Потомки выбранного корня и полный справочник для проверки дерева.

    Элементы списка содержат id/name/parent/path/is_leaf. ``records`` хранит
    все узлы исходного CSV, включая корень и другие предметы: сокращение
    списка кандидатов не должно превращать родителя в допустимый лист.
    """

    def __init__(self, rows, records, root_id):
        """Сохранить строки ветки, полную карту ID и идентификатор корня."""
        super().__init__(rows)
        self.records = records
        self.root_id = root_id


def normalize_catalog_id(value, *, allow_empty=False):
    """Нормализовать CSV ID без приближённого или строкового поиска.

    CSV-экспорт иногда содержит ``2.0`` вместо ``2`` и NULL у корней.
    Допускаются лишь целые десятичные значения; дробь, экспонента,
    отрицательное число и логическое значение не являются ID.
    """
    if value is None or (isinstance(value, str) and value.strip().casefold() in {"", "null", "none"}):
        if allow_empty:
            return 0
        raise ValueError("Пустой ID")
    if type(value) is int:
        if value < 0:
            raise ValueError("Отрицательный ID")
        return value
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.0+)?", value.strip()):
        raise ValueError(f"Некорректный ID: {value!r}")
    return int(value.strip().split(".", 1)[0])


def load_catalog(path, *, name_column, parent_column, root_id):
    """Прочитать CSV и вернуть ``Catalog`` потомков корня без самого корня.

    Названия колонок передаются явно, поскольку exams и topics имеют разные
    схемы. Для каждого потомка вычисляются полный путь внутри ветки и
    is_leaf; наличие детей проверяется по всем нормализованным ID в CSV.
    ValueError означает отсутствующие колонки/корень, неверные или повторные
    ID, пустое название, потерянного родителя, цикл либо пустую ветку.
    Повреждения вне выбранной ветки тоже отклоняются до обращения к модели.
    """
    records = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"id", name_column, parent_column}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Неверные колонки справочника: {path}")
        for row in reader:
            try:
                number = normalize_catalog_id(row["id"])
                parent = normalize_catalog_id(row[parent_column], allow_empty=True)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Некорректный ID или родитель в {path}: {row}") from error
            if number <= 0 or number in records:
                raise ValueError(f"Некорректный или повторный ID: {number}")
            name = " ".join((row[name_column] or "").split())
            if not name:
                raise ValueError(f"Пустое название категории {number}")
            records[number] = {
                "id": number,
                "name": name,
                "parent": parent,
            }

    if root_id not in records:
        raise ValueError(f"В справочнике нет корневой категории {root_id}")

    # Проверяем также сам корень и другие предметы: ошибка вне выбранной
    # ветки не должна скрывать повреждение общего справочника.
    paths = {}
    for number in records:
        current = number
        chain = []
        visited = set()
        while current and current not in paths:
            if current in visited:
                raise ValueError(f"Цикл в справочнике возле ID {current}")
            visited.add(current)
            if current not in records:
                raise ValueError(f"Не найден родитель категории: {current}")
            chain.append(current)
            current = records[current]["parent"]

        ancestors = paths.get(current, ())
        for ancestor in reversed(chain):
            ancestors = (*ancestors, ancestor)
            paths[ancestor] = ancestors

    parent_ids = {row["parent"] for row in records.values()}
    selected = []
    for number in sorted(records):
        path_ids = paths[number]
        if number == root_id or root_id not in path_ids:
            continue
        # Корень выбранной ветки не входит в подпись, как и прежде.
        relative_path = path_ids[path_ids.index(root_id) + 1:]
        selected.append({
            **records[number],
            "path": " > ".join(records[node]["name"] for node in relative_path),
            "is_leaf": number not in parent_ids,
        })
    if not selected:
        raise ValueError(f"Нет подразделов категории {root_id}")
    return Catalog(selected, records, root_id)


def allowed_categories(rows, prefix):
    """Вернуть только конечных потомков корня exams (E: 2) или topics (T: 1).

    С ``Catalog`` родительские связи проверяются по полному CSV, даже если
    кандидатов в rows осталось меньше. Лист может находиться на любой
    глубине. Тот же отбор используется в запросе и при проверке ответа/кэша.
    Неизвестный prefix вызывает ValueError; входной список не изменяется.
    """
    roots = {"E": 2, "T": 1}
    if prefix not in roots:
        raise ValueError(f"Неизвестный справочник: {prefix}")
    root_id = roots[prefix]
    # Дети определяются по полному CSV, даже если rows уже сокращён.
    records = getattr(rows, "records", {row["id"]: row for row in rows})
    parent_ids = {row["parent"] for row in records.values()}
    selected = []
    for row in rows:
        if row["id"] == root_id or row["id"] in parent_ids:
            continue
        current = row["parent"]
        visited = set()
        # Лист допустим, только если цепочка его родителей доходит до корня.
        while current not in (0, root_id):
            if current in visited or current not in records:
                break
            visited.add(current)
            current = records[current]["parent"]
        if current == root_id:
            selected.append(row)
    return selected


def build_category_choices(rows, prefix):
    """Вернуть отображение ``E/T | путь`` в ID для совместимого протокола.

    Включаются только допустимые листья. Полные пути сохраняют смысл темы,
    а совпадающие подписи получают различающие суффиксы. Эти подписи
    используются старым classify_tasks; classify_variant создаёт E/T-коды.
    """
    records = {row["id"]: row for row in rows}
    choices = {}

    for selected in sorted(allowed_categories(rows, prefix), key=lambda row: row["id"]):
        number = selected["id"]
        names = []
        current = number

        while current in records:
            row = records[current]
            names.append(row["name"])
            current = row["parent"]

        path = selected.get("path", " > ".join(reversed(names)))
        base_label = f"{prefix} | {path}"
        label = base_label
        duplicate = 2

        while label in choices:
            label = f"{base_label} [вариант {duplicate}]"
            duplicate += 1

        choices[label] = number

    return choices


def normalize_topic_name(value):
    """Сравнить названия без влияния регистра, ё, пробелов и пунктуации."""
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return " ".join(re.findall(r"[^\W_]+", value))


def resolve_new_topic(value, topics):
    """Вернуть ID совпавшего математического листа или новую тему строкой.

    Название нормализуется для поиска существующего листа. Неоднозначное
    совпадение, название родителя/чужой темы, заглушка и недопустимый формат
    вызывают ValueError. Осмысленное новое название остаётся строкой;
    семантическое отсутствие подходящей темы этот валидатор не доказывает.
    """
    if not isinstance(value, str):
        raise ValueError("Название новой темы должно быть строкой")
    name = " ".join(value.split())
    normalized = normalize_topic_name(name)
    placeholders = {
        "не найдено", "не найдена", "не найден", "другое", "другая тема",
        "нет темы", "нет подходящей темы", "тема отсутствует", "нет",
        "прочее", "неизвестно", "не определено", "новая тема",
        "unknown", "none", "null", "undefined", "n a", "other",
    }
    # Не принимаем и распространённые развёрнутые формы заглушек.
    # Например, слова «подходящей конечной» не превращают «нет темы»
    # в содержательное математическое название.
    placeholder = re.fullmatch(
        r"(?:тема )?(?:не найден[ао]?|не определен[ао]?)"
        r"(?: в (?:справочнике|каталоге))?"
        r"|(?:нет|не найден[ао]?) "
        r"(?:(?:подходящ(?:ей|ая)|конечн(?:ой|ая)|учебн(?:ой|ая)) )*"
        r"тем(?:ы|а)(?: в (?:справочнике|каталоге))?",
        normalized,
    )
    if (
        not normalized or normalized in placeholders or placeholder
        or not any(char.isalpha() for char in name)
    ):
        raise ValueError(f"Некорректное название новой темы: {value!r}")

    # Это может оказаться уже существующей темой. Её официальное название
    # вправе содержать числа или быть длиннее ограничения для новых названий.
    matches = [
        row for row in allowed_categories(topics, "T")
        if normalize_topic_name(row["name"]) == normalized
    ]
    if len(matches) > 1:
        raise ValueError(f"Неоднозначное название существующей темы: {name!r}")
    if matches:
        return matches[0]["id"]

    all_rows = getattr(topics, "records", {row["id"]: row for row in topics})
    if any(normalize_topic_name(row["name"]) == normalized for row in all_rows.values()):
        raise ValueError(f"Название совпадает с родительской или недопустимой темой: {name!r}")
    if (
        len(name) > 120 or len(name.split()) > 16
        or any(char.isnumeric() for char in name)
        or any(char in value for char in "\n\r\t<>|$\\`={}[]")
        # В том числе NUL, запрещённый в Excel/XML, и невидимые символы.
        or any(unicodedata.category(char).startswith("C") for char in value)
    ):
        raise ValueError(f"Некорректное название новой темы: {value!r}")
    return name


def decode_category_choices(items, exam_choices, topic_choices, topics=None):
    """Декодировать список записей совместимого протокола выбора категорий.

    На входе task_num/exam_choice/topic_kind/topic_choice, на выходе список
    task_num/exams_id/topics_id. Неизвестные подписи и неверная структура
    вызывают ValueError; темы двух справочников проверяются раздельно.
    Полноту набора задач затем проверяет validate_classification.
    """
    if not isinstance(items, list):
        raise ValueError("Ответ должен быть JSON-массивом")

    decoded = []

    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "task_num", "exam_choice", "topic_kind", "topic_choice"
        }:
            raise ValueError(
                "Нужны поля task_num, exam_choice, topic_kind, topic_choice"
            )

        record = {"task_num": item["task_num"]}

        label = item["exam_choice"]
        if not isinstance(label, str) or label not in exam_choices:
            raise ValueError(
                f"Задание {item['task_num']}: неизвестная подпись exam_choice={label!r}"
            )
        record["exams_id"] = exam_choices[label]

        if item["topic_kind"] == "existing":
            label = item["topic_choice"]
            if not isinstance(label, str) or label not in topic_choices:
                raise ValueError(f"Задание {item['task_num']}: неизвестная подпись topic_choice={label!r}")
            record["topics_id"] = topic_choices[label]
        elif item["topic_kind"] == "new":
            if topics is None:
                raise ValueError("Для новой темы нужен полный справочник topics")
            record["topics_id"] = resolve_new_topic(item["topic_choice"], topics)
        else:
            raise ValueError(f"Недопустимый topic_kind: {item['topic_kind']!r}")

        decoded.append(record)

    return decoded


def validate_classification(items, exams, topics, *, task_numbers=None):
    """Проверить ID и вернуть ``{task_num: {exams_id, topics_id}}``.

    Каждый ожидаемый номер должен встретиться ровно один раз. По умолчанию
    ожидается старый набор 1–19; task_numbers задаёт набор текущего варианта.
    Числовые ID обязаны быть листьями своей ветки, новая строковая тема
    проходит resolve_new_topic. Ошибки контракта вызывают ValueError.
    Математический смысл формально допустимого выбора требует сверки.
    """
    expected_numbers = TASK_NUMBERS if task_numbers is None else set(task_numbers)
    if not isinstance(items, list):
        raise ValueError("Классификация должна быть JSON-массивом")
    exams_ids = {row["id"] for row in allowed_categories(exams, "E")}
    topics_ids = {row["id"] for row in allowed_categories(topics, "T")}
    result = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "task_num", "exams_id", "topics_id"
        }:
            raise ValueError("В записи нужны только task_num, exams_id, topics_id")
        number = item["task_num"]
        if type(number) is not int or number not in expected_numbers:
            raise ValueError(f"Некорректный номер задания: {number}")
        if number in result:
            raise ValueError(f"Повторная классификация задания {number}")
        exam_id = item["exams_id"]
        topic = item["topics_id"]
        if type(exam_id) is not int or exam_id not in exams_ids:
            raise ValueError(f"Задание {number}: недопустимый exams_id={exam_id!r}")
        if isinstance(topic, str):
            topic = resolve_new_topic(topic, topics)
        elif type(topic) is not int or topic not in topics_ids:
            raise ValueError(f"Задание {number}: недопустимый topics_id={topic!r}")
        result[number] = {
            "exams_id": exam_id,
            "topics_id": topic,
        }
    if set(result) != expected_numbers:
        description = "1–19" if task_numbers is None else ", ".join(map(str, sorted(expected_numbers)))
        raise ValueError(f"Не все задания {description} получили классификацию")
    return result
