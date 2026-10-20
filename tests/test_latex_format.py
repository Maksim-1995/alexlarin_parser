"""Проверки оформления обозначений без изменения математического содержания."""

import unittest

from latex_format import normalize_condition


class LatexFormatTests(unittest.TestCase):
    def test_geometry_and_missing_space(self):
        source = (
            "В остроугольном треугольнике ABC высоты AA_1 и CC_1 "
            "пересекаются в точке H. Радиус описанной около треугольника "
            "ABC окружности равен10. Расстояние от центра описанной "
            "окружности до стороны AC равно 6. Найдите длину отрезка BH."
        )
        expected = (
            "В остроугольном треугольнике $ABC$ высоты $AA_1$ и $CC_1$ "
            "пересекаются в точке $H$. Радиус описанной около треугольника "
            "$ABC$ окружности равен $10$. Расстояние от центра описанной "
            "окружности до стороны $AC$ равно $6$. Найдите длину отрезка $BH$."
        )
        self.assertEqual(normalize_condition(source), expected)

    def test_vectors_remain_latex(self):
        source = (
            r"Векторы $\vec{a}$ и $\vec{b}$ имеют длины $|\vec{a}|=3$ "
            r"и $|\vec{b}|=4$ . Известно, что длина вектора $2\vec{a}-\vec{b}$ "
            r"равна $\sqrt{43}$ . Найдите скалярное произведение векторов "
            r"$\vec{a}$ и $\vec{b}$ ."
        )
        self.assertEqual(normalize_condition(source), source.replace("$ .", "$."))

    def test_power_and_logarithm_unchanged(self):
        source = (
            r"Решите уравнение $x^{\log_3 x} = 81x^3$. Если корней несколько, "
            "в ответе укажите их произведение."
        )
        self.assertEqual(normalize_condition(source), source)

    def test_parameter_system_unchanged(self):
        source = (
            r"Найдите все значения параметра $a$, при каждом из которых система "
            r"$\begin{cases} \frac{\log_{|x|}(x^2-2x-a)-\log_{|x|}(2|x-1|+a)}"
            r"{\sqrt{7x-4a}}=0 \\ x^2-3x-4\le0 \end{cases}$ "
            "имеет единственное решение."
        )
        self.assertEqual(normalize_condition(source), source)

    def test_repeated_processing_does_not_add_delimiters(self):
        once = normalize_condition("Высота AA_1 равна 3,5. Найдите BH.")
        self.assertEqual(normalize_condition(once), once)

    def test_braced_and_multiple_indices_are_wrapped_as_one_label(self):
        self.assertEqual(
            normalize_condition("Отрезки AA_{12} и A_1B_1."),
            "Отрезки $AA_{12}$ и $A_1B_1$.",
        )

    def test_display_math_and_math_with_newlines_are_preserved(self):
        source = "Формула $$x^2 + y^2\n= 25$$. Ещё $x\n+1$."
        self.assertEqual(normalize_condition(source), source)

    def test_latin_subpart_markers_stay_plain(self):
        self.assertEqual(
            normalize_condition("A) Найдите AC. B) Найдите BH."),
            "A) Найдите $AC$. B) Найдите $BH$.",
        )

    def test_broken_math_is_reported_before_replacements(self):
        for source in ("Найдите $x + 1.", "Формула $$x + 1$.", r"Корень \sqrt{43}."):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    normalize_condition(source)

    def test_escaped_dollar_inside_math_is_preserved(self):
        source = r"Формула $\text{цена \$} + x$."
        self.assertEqual(normalize_condition(source), source)

    def test_literal_newlines_before_subparts(self):
        source = r"Пирамида SABCD.\nA) Найдите AC.\nБ) Найдите объем."
        expected = "Пирамида $SABCD$.\nA) Найдите $AC$.\nБ) Найдите объем."

        self.assertEqual(normalize_condition(source), expected)

    def test_math_commands_are_preserved(self):
        source = r"Формула $\nu + \nabla f \neq 0$.\nБ) Найдите $a$."
        expected = "Формула $\\nu + \\nabla f \\neq 0$.\nБ) Найдите $a$."

        self.assertEqual(normalize_condition(source), expected)


if __name__ == "__main__":
    unittest.main()
