"""Извлечение условий заданий и ответов из материалов вариантов ЕГЭ."""

import re

TASK_RE = re.compile(
    r"^[ \t]*(1[0-9]|[1-9])\.(?=[ \t]|$)",
    flags=re.MULTILINE,
)

ANSWER_RE = re.compile(
    r'getElementById\(\s*["\']T(\d+)["\']\s*\)\.value\s*=\s*'
    r'["\']([^"\']+)["\']'
)


def clean_condition(text):
    """Очистить условие задания от служебного текста и лишних пробелов.

    Args:
        text (str): Исходный текст условия.

    Returns:
        str: Условие без поля ответа, колонтитулов и завершающих
            инструкций, с обрезанными пробелами в строках и не более чем
            одной пустой строкой подряд.
    """
    # Удаляем пустое поле для записи ответа.
    text = re.sub(
        r"Ответ:[ \t]*_+\.?",
        "",
        text,
    )

    # Удаляем строки с колонтитулами.
    text = re.sub(
        r"^.*(?:Единый государственный экзамен,|alexlarin\.net|"
        r"Разрешается свободное копирование).*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Отсекаем инструкции, попавшие после условия.
    text = re.split(
        r"^[ \t]*(?:Не забудьте перенести|Часть[ \t]+2\b|Проверьте, чтобы)",
        text,
        maxsplit=1,
        flags=re.MULTILINE,
    )[0]

    # Удаляем пробелы по краям каждой строки.
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    # Три и более переноса заменяем двумя.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_tasks(text):
    """Извлечь задания 1–19 из текста варианта в порядке их номеров.

    Args:
        text (str): Полный текст варианта с нумерованными заданиями.

    Returns:
        list[dict]: Словари с номером задания ``task_num`` и очищенным
            условием ``condition``.

    Raises:
        ValueError: Если номера не образуют последовательность 1–19
            или после очистки найдено пустое условие.
    """
    matches = list(TASK_RE.finditer(text))
    numbers = [int(match.group(1)) for match in matches]

    if numbers != list(range(1, 20)):
        raise ValueError(
            f"Ожидались задания 1–19 по порядку. Получены: {numbers}"
        )

    tasks = []

    for index, match in enumerate(matches):
        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        condition = clean_condition(text[start:end])
        task_num = int(match.group(1))

        if not condition:
            raise ValueError(
                f"Пустое условие задания {task_num}"
            )

        tasks.append({
            "task_num": task_num,
            "condition": condition,
        })

    return tasks


def parse_js_answers(js_text):
    """Извлечь ответы на задания 1–12 из JavaScript-кода варианта.

    Args:
        js_text (str): Код с присваиваниями значения полям T1–T12
            через ``getElementById``.

    Returns:
        dict[int, str]: Ответы, сопоставленные номерам заданий.

    Raises:
        ValueError: Если найдены не все номера 1–12, повторы номеров
            или ответы с другими номерами.
    """
    matches = ANSWER_RE.findall(js_text)

    numbers = sorted(
        int(number)
        for number, value in matches
    )

    if numbers != list(range(1, 13)):
        raise ValueError(
            "Ожидались ответы на задания 1–12 без повторов"
        )

    answers = {}

    for number, value in matches:
        answers[int(number)] = value

    return answers
