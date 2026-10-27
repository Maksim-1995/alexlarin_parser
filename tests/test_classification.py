import copy
import json
import tempfile
import unittest
from pathlib import Path

from classification import classify_tasks, load_catalog, validate_classification


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "classification.json"
        self.tasks = [
            {"task_num": n, "condition": "Найдите скалярное произведение векторов."}
            for n in range(1, 20)
        ]
        self.labels = [
            {"task_num": n, "exams_id": 180, "topics_id": 526}
            for n in range(1, 20)
        ]
        self.exam_label = "E180 | Вектор > Векторы и операции с ними"
        self.topic_label = (
            "T526 | Геометрия > Планиметрия > Векторы > "
            "Умножение вектора на число. Скалярное произведение векторов"
        )
        self.response_items = [
            {
                "task_num": n,
                "exam_choice": self.exam_label,
                "topic_choice": self.topic_label,
            }
            for n in range(1, 20)
        ]


    def classify(self, request, **kwargs):
        return classify_tasks(
            self.tasks,
            DATA_DIR / "exams.csv",
            DATA_DIR / "topics.csv",
            self.cache,
            request=request,
            **kwargs,
        )

    def test_all_nineteen_are_sent_once_and_input_is_unchanged(self):
        calls = []
        original = copy.deepcopy(self.tasks)

        def request(prompt, *, model):
            calls.append(prompt)
            payload = json.loads(prompt.split("Данные:\n", 1)[1])
            self.assertEqual(len(payload["tasks"]), 19)
            self.assertTrue(payload["exams"] and payload["topics"])
            return json.dumps(self.response_items)

        result = self.classify(request)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result), 19)
        self.assertEqual(result[2], {"exams_id": 180, "topics_id": 526})
        self.assertEqual(self.tasks, original)

    def test_matching_cache_makes_no_request(self):
        self.classify(lambda *args, **kwargs: json.dumps(self.response_items))

        def unexpected_request(*args, **kwargs):
            self.fail("При подходящем кэше запрос не нужен")

        result = self.classify(unexpected_request)
        self.assertEqual(len(result), 19)

    def test_invalid_choice_is_repaired_by_second_request(self):
        wrong = copy.deepcopy(self.response_items)
        wrong[0]["exam_choice"] = self.topic_label
        calls = []

        def request(prompt, *, model):
            calls.append(prompt)
            return json.dumps(
                wrong if len(calls) == 1 else self.response_items
            )

        self.classify(request)

        self.assertEqual(len(calls), 2)
        self.assertIn("неизвестная подпись exam_choice", calls[1])

        cached = json.loads(self.cache.read_text(encoding="utf-8"))
        self.assertEqual(cached["requests_used"], 2)

        def request(prompt, *, model):
            calls.append(prompt)
            return json.dumps(wrong if len(calls) == 1 else self.labels)

        self.classify(request)
        self.assertEqual(len(calls), 2)
        self.assertIn("неизвестная подпись exam_choice", calls[1])
        cached = json.loads(self.cache.read_text(encoding="utf-8"))
        self.assertEqual(cached["requests_used"], 2)

    def test_two_bad_responses_stop_without_saving_cache(self):
        calls = []

        def request(prompt, *, model):
            calls.append(prompt)
            return "не JSON"

        with self.assertRaisesRegex(ValueError, "после двух запросов"):
            self.classify(request)
        self.assertEqual(len(calls), 2)
        self.assertFalse(self.cache.exists())

    def test_changed_condition_invalidates_cache(self):
        self.classify(lambda *args, **kwargs: json.dumps(self.response_items))
        self.tasks[0]["condition"] += " Длины векторов заданы."
        calls = []

        def request(prompt, *, model):
            calls.append(prompt)
            return json.dumps(self.response_items)

        self.classify(request)
        self.assertEqual(len(calls), 1)

    def test_network_error_does_not_start_hidden_application_retry(self):
        calls = []

        def request(prompt, *, model):
            calls.append(prompt)
            raise TimeoutError("Тестовый тайм-аут")

        with self.assertRaises(TimeoutError):
            self.classify(request)
        self.assertEqual(len(calls), 1)
        self.assertFalse(self.cache.exists())

    def test_missing_task_is_rejected_before_request(self):
        self.tasks.pop()
        with self.assertRaisesRegex(ValueError, "ровно задания 1–19"):
            self.classify(lambda *args, **kwargs: self.fail("Запрос запрещён"))

    def test_validation_rejects_multiple_ids_duplicate_and_missing_tasks(self):
        exams = load_catalog(
            DATA_DIR / "exams.csv", name_column="name_of_Exam",
            parent_column="parent_of_exam", root_id=2,
        )
        topics = load_catalog(
            DATA_DIR / "topics.csv", name_column="name",
            parent_column="parent", root_id=1,
        )
        wrong_multiple = copy.deepcopy(self.labels)
        wrong_multiple[0]["topics_id"] = [499, 526]
        wrong_duplicate = copy.deepcopy(self.labels)
        wrong_duplicate[-1]["task_num"] = 1
        wrong_boolean = copy.deepcopy(self.labels)
        wrong_boolean[0]["topics_id"] = True
        for bad in (wrong_multiple, wrong_duplicate, wrong_boolean, self.labels[:-1]):
            with self.subTest(response=bad[0]):
                with self.assertRaises(ValueError):
                    validate_classification(bad, exams, topics)


    def test_cache_stores_numeric_ids_and_selected_labels(self):
        self.classify(
            lambda *args, **kwargs: json.dumps(self.response_items)
        )
        cached = json.loads(self.cache.read_text(encoding="utf-8"))

        self.assertEqual(cached["tasks"], self.labels)
        self.assertEqual(cached["choices"], self.response_items)


if __name__ == "__main__":
    unittest.main()
