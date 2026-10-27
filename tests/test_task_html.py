import unittest

from task_html import add_subtask_breaks


class TaskHtmlTests(unittest.TestCase):
    def test_common_condition_and_each_subpart_start_on_new_line(self):
        self.assertEqual(
            add_subtask_breaks("Дана пирамида. А) Докажите. Б) Найдите объём."),
            "Дана пирамида.<br>А) Докажите.<br>Б) Найдите объём.",
        )

    def test_no_leading_break_when_first_part_starts_the_condition(self):
        self.assertEqual(
            add_subtask_breaks("а) Решите уравнение. б) Отберите корни."),
            "а) Решите уравнение.<br>б) Отберите корни.",
        )

    def test_real_and_literal_newlines_are_converted_without_duplicates(self):
        self.assertEqual(
            add_subtask_breaks("Условие.\nА) Первое.\\nБ) Второе."),
            "Условие.<br>А) Первое.<br>Б) Второе.",
        )

    def test_latex_is_preserved_including_cases_and_labels_inside_formulas(self):
        formula = r"$\begin{cases}f(a)=1\\\text{а) проверка}\end{cases}$"
        self.assertEqual(
            add_subtask_breaks("А) " + formula + " Б) Найдите $a$"),
            "А) " + formula + "<br>Б) Найдите $a$",
        )

    def test_existing_breaks_and_repeated_formatting_are_idempotent(self):
        source = "Условие.<BR />\nА) Первое.<br>Б) Второе."
        expected = "Условие.<br>А) Первое.<br>Б) Второе."
        result = add_subtask_breaks(source)
        self.assertEqual(result, expected)
        self.assertEqual(add_subtask_breaks(result), result)

    def test_words_and_plain_function_arguments_are_not_subpart_labels(self):
        text = "Это координата), а f(a) — обозначение. Найдите $x$."
        self.assertEqual(add_subtask_breaks(text), text)

    def test_unclosed_formula_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Непарный"):
            add_subtask_breaks("А) Найдите $x. Б) Продолжите.")


if __name__ == "__main__":
    unittest.main()
