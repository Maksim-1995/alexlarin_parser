"""Переносы подпунктов в HTML с сохранением формул LaTeX."""

import re

from latex_format import MATH_RE


# Оформление LaTeX и HTML должно одинаково определять границы формул:
# экранированный \$ в обычном тексте не начинает математический блок.
LABEL_RE = re.compile(r"(?:(?<!\S)|(?<=[.!?;:])|(?<=<br>))([АБВГABCDабвгabcd]\))")


def add_subtask_breaks(text):
    """Добавить <br> перед подпунктами вне формул; повторный вызов безопасен."""
    if not isinstance(text, str):
        raise ValueError("Для HTML-оформления нужна строка")

    result = ""
    for index, part in enumerate(MATH_RE.split(text)):
        if index % 2:
            result += part
            continue
        if re.search(r"(?<!\\)\$", part):
            raise ValueError("Непарный знак $ при оформлении подпунктов")
        part = re.sub(r"<br\s*/?>", "<br>", part, flags=re.IGNORECASE)
        part = re.sub(r"\\n(?=\s*[АБВГABCDабвгabcd]\))", "\n", part)
        previous = 0
        for match in LABEL_RE.finditer(part):
            result += part[previous:match.start()]
            result = result.rstrip()
            if result and not result.endswith("<br>"):
                result += "<br>"
            result += match.group(1)
            previous = match.end()
        result += part[previous:]
    return result
