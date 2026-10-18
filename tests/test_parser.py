"""Проверки разбора условий и нумерации заданий."""

import pytest

from parsers import parse_tasks


def make_text(numbers=range(1, 20)):
    """Собрать текст заданий с указанными номерами."""
    return "\n\n".join(
        f"{number}. Условие задания {number}."
        for number in numbers
    )


def test_parse_tasks_preserves_multiline_condition():
    """Проверить сохранение строк и абзацев многострочного условия."""
    text = make_text().replace(
        "1. Условие задания 1.",
        "1. Найдите сумму.\n"
        "2.5 + 3.5\n\n"
        "Запишите результат.\n"
        "Ответ: _______.",
    )

    tasks = parse_tasks(text)

    assert len(tasks) == 19

    assert tasks[0] == {
        "task_num": 1,
        "condition": (
            "Найдите сумму.\n"
            "2.5 + 3.5\n\n"
            "Запишите результат."
        ),
    }

    assert tasks[-1] == {
        "task_num": 19,
        "condition": "Условие задания 19.",
    }


@pytest.mark.parametrize(
    "numbers",
    [
        [number for number in range(1, 20) if number != 7],
        list(range(1, 20)) + [19],
    ],
    ids=["missing_number", "duplicate_number"],
)
def test_parse_tasks_rejects_invalid_numbers(numbers):
    """Проверить отклонение пропущенных и повторяющихся номеров."""
    text = make_text(numbers)

    with pytest.raises(ValueError, match="Ожидались задания"):
        parse_tasks(text)


def test_parse_tasks_rejects_empty_condition():
    """Проверить отклонение задания с пустым условием."""
    text = make_text().replace(
        "5. Условие задания 5.",
        "5. Ответ: _______.",
    )

    with pytest.raises(ValueError, match="Пустое условие задания 5"):
        parse_tasks(text)
