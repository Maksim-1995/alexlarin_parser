"""Проверки сохранения структуры формул при выводе в текстовые ячейки."""

import unittest

from math_text import condition_to_text


class MathTextTests(unittest.TestCase):
    def test_vectors_and_root(self):
        self.assertEqual(
            condition_to_text(r"Векторы $\vec{a}$ и $\vec{b}$; $|\vec{a}|=3$; $\sqrt{43}$ ."),
            "Векторы a⃗ и b⃗; |a⃗|=3; √43.",
        )

    def test_parameter_has_no_markup(self):
        self.assertEqual(condition_to_text("Найдите значения $a$."), "Найдите значения a.")

    def test_fraction_keeps_both_groups(self):
        self.assertEqual(
            condition_to_text(r"$\frac{x+1}{x-2}$"), "(x+1)/(x-2)"
        )

    def test_nested_fraction_and_root(self):
        self.assertEqual(
            condition_to_text(r"$\frac{1}{\sqrt{\frac{x+1}{x-2}}}$"),
            "1/(√((x+1)/(x-2)))",
        )

    def test_compound_power_keeps_its_extent(self):
        self.assertEqual(
            condition_to_text(r"$x^{\log_3 x}=81x^3$"), "x^(log₃ x)=81x³"
        )

    def test_absolute_value_base_remains_grouped(self):
        self.assertEqual(
            condition_to_text(r"$\log_{|x|}(x^2-2x-a)$"), "log[|x|](x²-2x-a)"
        )

    def test_system_preserves_the_two_equations(self):
        self.assertEqual(
            condition_to_text(
                r"$\begin{cases}\frac{x-a}{\sqrt{7x-4a}}=0\\x^2-3x-4\le0\end{cases}$"
            ),
            "{\n(x-a)/(√(7x-4a))=0\nx²-3x-4≤0\n}",
        )

    def test_subscripts_degrees_and_percent(self):
        self.assertEqual(
            condition_to_text(r"$T_0=25^{\circ}C$; $4\%$; AA_1; $\omega_2$"),
            "T₀=25°C; 4%; AA₁; ω₂",
        )

    def test_unknown_or_incomplete_notation_is_not_discarded(self):
        for source in (
            r"$\unknown{x}$", r"$\sqrt{x$", r"$\begin{cases}x=1$", "$a",
            r"$\sqrt{}$", r"$\frac{}{2}$",
        ):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    condition_to_text(source)

    def test_fraction_followed_by_a_factor_stays_grouped(self):
        self.assertEqual(condition_to_text(r"$\frac{1}{2}x$"), "(1/2)x")

    def test_power_of_a_fraction_or_root_applies_to_the_whole_expression(self):
        self.assertEqual(condition_to_text(r"$\frac{x}{y}^2$"), "(x/y)²")
        self.assertEqual(condition_to_text(r"$\sqrt{x}^2$"), "(√x)²")

    def test_unbraced_root_argument_does_not_absorb_next_digit(self):
        self.assertEqual(condition_to_text(r"$\sqrt43$"), "√(4)3")


if __name__ == "__main__":
    unittest.main()
