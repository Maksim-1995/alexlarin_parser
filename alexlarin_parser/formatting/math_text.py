"""Читаемая линейная запись формул для текстовых ячеек Excel.

Это преобразование оформления, без решения или упрощения выражений.
Поддерживаются команды, используемые в школьных условиях этого варианта.
Неизвестная команда вызывает ошибку вместо потери части формулы.
Полное оформление дробей и степеней, как в PDF, текстовая ячейка не сохраняет.
"""

import re


SUPERSCRIPT = dict(zip("0123456789+-=()nxabkt", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿˣᵃᵇᵏᵗ"))
SUBSCRIPT = dict(zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))

SYMBOLS = {
    "pi": "π", "omega": "ω", "alpha": "α", "beta": "β",
    "gamma": "γ", "delta": "δ", "theta": "θ", "lambda": "λ",
    "mu": "μ", "nu": "ν", "sigma": "σ", "phi": "φ", "nabla": "∇",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠",
    "in": "∈", "notin": "∉", "infty": "∞", "cup": "∪", "cap": "∩",
    "cdot": "·", "times": "×", "div": "÷", "pm": "±", "circ": "°",
    "lvert": "|", "rvert": "|", "vert": "|", "lVert": "‖", "rVert": "‖",
    "quad": " ", "qquad": " ", "colon": ":", "ldots": "…", "dots": "…",
}
FUNCTIONS = {"sin", "cos", "tan", "tg", "cot", "ctg", "ln", "exp", "arcsin", "arccos"}
TEXT_COMMANDS = {"text", "textrm", "mathrm", "mathit", "operatorname"}


def _script(text, alphabet, fallback):
    text = text.strip()
    if not text:
        raise ValueError("Пустой показатель степени или индекс")
    if all(char in alphabet for char in text):
        return "".join(alphabet[char] for char in text)
    return f"{fallback}({text})"


def _fraction_part(text):
    """Скобки сохраняют границы числителя и знаменателя."""
    text = text.strip()
    if re.fullmatch(r"[0-9]+|[^\W\d_]", text, flags=re.UNICODE):
        return text
    return f"({text})"


class _Reader:
    def __init__(self, text):
        self.text = text
        self.position = 0
        self.environments = []
        self.math_open = False

    def following(self):
        return self.text[self.position:].lstrip()

    def sequence(self, stop=None):
        parts = []
        while self.position < len(self.text):
            char = self.text[self.position]
            self.position += 1
            if char == "}":
                if stop == "}":
                    return "".join(parts)
                raise ValueError("Закрывающая скобка LaTeX без открывающей")
            if char == "{":
                parts.append(self.sequence(stop="}"))
            elif char == "\\":
                parts.append(self.command())
            elif char == "$":
                self.math_open = not self.math_open
            elif char == "^":
                exponent = self.argument().strip()
                parts.append("°" if exponent == "°" else _script(exponent, SUPERSCRIPT, "^"))
            elif char == "_":
                parts.append(_script(self.argument(), SUBSCRIPT, "_"))
            elif char == "&":
                parts.append(" ")  # Разделитель столбцов в записи системы.
            else:
                parts.append(char)
        if stop is not None:
            raise ValueError("Незакрытая фигурная скобка LaTeX")
        return "".join(parts)

    def argument(self):
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1
        if self.position >= len(self.text):
            raise ValueError("У команды LaTeX отсутствует аргумент")
        char = self.text[self.position]
        self.position += 1
        if char == "{":
            return self.sequence(stop="}")
        if char == "\\":
            return self.command()
        if char in "}$^_":
            raise ValueError("Некорректный аргумент команды LaTeX")
        return char

    def command(self):
        if self.position >= len(self.text):
            raise ValueError("Обратный слеш без команды LaTeX")
        match = re.match(r"[A-Za-z]+", self.text[self.position:])
        if match is None:
            symbol = self.text[self.position]
            self.position += 1
            if symbol == "\\":
                return "\n"
            if symbol in "()[]":  # Разделители формул \(...\) и \[...\].
                return ""
            if symbol in ",;: ":
                return " "
            if symbol == "!":
                return ""
            if symbol in "{}%_$&#|":
                return symbol
            raise ValueError(f"Не поддерживается команда LaTeX: \\{symbol}")

        name = match.group()
        self.position += len(name)
        if name in SYMBOLS:
            return SYMBOLS[name]
        if name in FUNCTIONS:
            return name
        if name in {"left", "right", "big", "Big", "bigl", "bigr"}:
            return ""
        if name in TEXT_COMMANDS:
            return self.argument()
        if name == "mathbb":
            body = self.argument()
            return {"R": "ℝ", "Z": "ℤ", "N": "ℕ", "Q": "ℚ", "C": "ℂ"}.get(body, body)
        if name in {"vec", "overrightarrow"}:
            body = self.argument().strip()
            if not body:
                raise ValueError("Пустое обозначение вектора")
            return body + "\u20d7" if len(body) == 1 else f"→({body})"
        if name == "sqrt":
            body = self.argument().strip()
            if not body:
                raise ValueError("Пустое подкоренное выражение")
            # У сложного подкоренного выражения нужны явные скобки.
            simple = re.fullmatch(r"[0-9]+|[^\W\d_]", body, flags=re.UNICODE)
            following = self.following()
            adjacent = bool(following) and following[0].isalnum()
            root = "√" + (body if simple and not adjacent else f"({body})")
            if following.startswith(("^", "_")):
                root = f"({root})"
            return root
        if name in {"frac", "dfrac", "tfrac"}:
            numerator = self.argument()
            denominator = self.argument()
            if not numerator.strip() or not denominator.strip():
                raise ValueError("Пустой числитель или знаменатель")
            fraction = f"{_fraction_part(numerator)}/{_fraction_part(denominator)}"
            # Например, \frac{1}{2}x должно остаться (1/2)x.
            following = self.following()
            if following and (following[0].isalnum() or following[0] in "^_"):
                fraction = f"({fraction})"
            elif following.startswith("\\") and not following.startswith(("\\right", "\\end", "\\\\")):
                fraction = f"({fraction})"
            return fraction
        if name == "log":
            saved_position = self.position
            while self.position < len(self.text) and self.text[self.position].isspace():
                self.position += 1
            if self.position < len(self.text) and self.text[self.position] == "_":
                self.position += 1
                base = self.argument().strip()
                if base and all(char in SUBSCRIPT for char in base):
                    return "log" + _script(base, SUBSCRIPT, "_")
                return f"log[{base}]"
            self.position = saved_position
            return "log"
        if name in {"begin", "end"}:
            environment = self.argument().strip()
            if environment != "cases":
                raise ValueError(f"Не поддерживается окружение LaTeX: {environment}")
            if name == "begin":
                self.environments.append(environment)
                return "{\n"
            if not self.environments or self.environments.pop() != environment:
                raise ValueError("Нарушена структура системы LaTeX")
            return "\n}"
        raise ValueError(f"Не поддерживается команда LaTeX: \\{name}")


def condition_to_text(condition):
    """Преобразовать поддерживаемую запись, сохраняя группировку выражений."""
    condition = re.sub(r"\\n(?=\s*[АБВABC]\))", "\n", condition)
    reader = _Reader(condition)
    result = reader.sequence()
    if reader.math_open or reader.environments:
        raise ValueError("Незакрытая формула или система LaTeX")
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r" +([.,;:!?])", r"\1", result)
    result = re.sub(r" *\n *", "\n", result)
    return result.strip()
