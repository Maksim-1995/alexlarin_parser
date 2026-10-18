"""Проверки извлечения ответов из JavaScript-кода."""

import pytest

from parsers import parse_js_answers

JS_TEXT = """
document.getElementById("T1").value = "12";
document.getElementById("T2").value = "2,25";
document.getElementById("T3").value = "10,125";
document.getElementById("T4").value = "0,3744";
document.getElementById("T5").value = "0,9940";
document.getElementById("T6").value = "27";
document.getElementById("T7").value = "0";
document.getElementById("T8").value = "12";
document.getElementById("T9").value = "3";
document.getElementById("T10").value = "5";
document.getElementById("T11").value = "-2";
document.getElementById("T12").value = "1";
"""


def test_answers_preserve_text_format():
    """Проверить сохранение запятых, конечных нулей и знака ответа."""
    answers = parse_js_answers(JS_TEXT)

    assert len(answers) == 12
    assert answers[5] == "0,9940"
    assert answers[7] == "0"
    assert answers[11] == "-2"


def test_answers_ignore_comparison():
    """Проверить пропуск сравнения значения поля с ответом."""
    comparison = (
        'if (document.getElementById("T1").value == "999") {}\n'
    )

    answers = parse_js_answers(comparison + JS_TEXT)

    assert answers[1] == "12"


def test_answers_reject_missing_answer():
    """Проверить отклонение набора с пропущенным ответом."""
    js_text = JS_TEXT.replace(
        'document.getElementById("T12").value = "1";',
        "",
    )

    with pytest.raises(ValueError, match="Ожидались ответы"):
        parse_js_answers(js_text)


def test_answers_reject_duplicate_answer():
    """Проверить отклонение набора с повторяющимся номером ответа."""
    js_text = (
        JS_TEXT
        + '\ndocument.getElementById("T1").value = "999";'
    )

    with pytest.raises(ValueError, match="Ожидались ответы"):
        parse_js_answers(js_text)
