"""Подготовка условий с LaTeX для проверки через MathJax.

Готовые формулы в $...$ и $$...$$ сохраняются без изменений.
В обычном тексте оформляются латинские обозначения и числа.
Функция не восстанавливает формулы, ошибочно прочитанные OCR.
"""

import re


MATH_RE = re.compile(
    r"("
    r"(?<!\\)\$\$(?:\\.|[^\\])*?\$\$"
    r"|(?<![\\$])\$(?!\$)(?:\\.|[^\\$])*\$(?!\$)"
    r")",
    flags=re.DOTALL,
)

TOKEN_RE = re.compile(
    r"(?<![\w\\])(?:[A-Z](?:_(?:\d+|\{\d+\}))?)+(?!\w)"
    r"|(?<![\w\\])\d+(?:[.,]\d+)?(?!\w)"
)


def _wrap_token(match):
    """Оформить обозначение или число, сохранив метки подпунктов A), B), C)."""
    token = match.group()
    following = match.string[match.end():]
    if token in {"A", "B", "C"} and following.startswith(")"):
        return token
    return f"${token}$"


def normalize_condition(condition):
    """Обработать обычный текст, не меняя содержимое готовых формул."""
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError("Условие должно быть непустой строкой")

    # Скобки в MATH_RE сохраняют найденные формулы в списке частей.
    # Чётные позиции — обычный текст, нечётные — готовые формулы.
    parts = MATH_RE.split(condition)

    for index in range(0, len(parts), 2):
        text = parts[index]
        text = re.sub(r"\\n(?=\s*[АБВABC]\))", "\n", text)
        if re.search(r"(?<!\\)\$", text):
            raise ValueError("Непарный знак $: проверьте границы формулы")
        if re.search(r"\\[A-Za-z]+|\\[()\[\]]", text):
            raise ValueError(
                "Команда LaTeX вне $...$: проверьте границы формулы"
            )

        # Например: «равен10» -> «равен 10».
        text = re.sub(r"([а-яё]{2,})(?=\d)", r"\1 ", text)
        text = TOKEN_RE.sub(_wrap_token, text)
        text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
        parts[index] = text

    return "".join(parts).strip()
