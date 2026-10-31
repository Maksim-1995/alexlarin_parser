"""Проверки OCR без сетевых запросов и расходования API-квоты."""

import hashlib
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pymupdf

import recognizer


class RecognizerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pdf_path = self.root / "variant.pdf"
        self.pdf_path.write_bytes(b"%PDF-test-input")
        self.cache_path = self.root / "variant_conditions_gemini.json"
        self.stdout = StringIO()
        self.silent = redirect_stdout(self.stdout)
        self.silent.__enter__()
        self.addCleanup(self.silent.__exit__, None, None, None)

    def state(self, numbers):
        return {
            "version": recognizer.CACHE_VERSION,
            "method": recognizer.METHOD,
            "pdf_sha256": hashlib.sha256(self.pdf_path.read_bytes()).hexdigest(),
            "model": recognizer.MODEL,
            "prompt": recognizer.PROMPT,
            "dpi": recognizer.DPI,
            "tasks": [{"task_num": n, "condition": f"Условие {n}"} for n in numbers],
        }

    def save(self, state):
        self.cache_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    def fake_images(self, pdf_path, image_dir):
        image_dir.mkdir(parents=True, exist_ok=True)
        images = {}
        for number in range(1, 20):
            image_path = image_dir / f"task_{number:02d}.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n-test-image")
            images[number] = image_path
        return images

    @staticmethod
    def response(**request):
        number = int(request["contents"][0].rsplit(" ", 1)[1])
        return SimpleNamespace(text=json.dumps([
            {"task_num": number, "condition": f"Новое условие {number}"}
        ], ensure_ascii=False))

    def test_full_cache_needs_neither_api_key_nor_images(self):
        self.save(self.state(range(1, 20)))
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(recognizer.genai, "Client") as client,
            patch.object(recognizer, "render_task_images") as render,
        ):
            result = recognizer.recognize_conditions(self.pdf_path)
        self.assertEqual(sorted(result), list(range(1, 20)))
        client.assert_not_called()
        render.assert_not_called()
        self.assertIn("Запросов OCR: 0.", self.stdout.getvalue())

    def test_interrupted_run_resumes_without_repeating_saved_tasks(self):
        requested = []

        def interrupted(**request):
            number = int(request["contents"][0].rsplit(" ", 1)[1])
            requested.append(number)
            if number == 3:
                raise TimeoutError("Имитируем сетевой сбой")
            return self.response(**request)

        client = MagicMock()
        client.models.generate_content.side_effect = interrupted
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.object(recognizer, "render_task_images", side_effect=self.fake_images),
            patch.object(recognizer.genai, "Client") as factory,
        ):
            factory.return_value.__enter__.return_value = client
            with self.assertRaises(TimeoutError):
                recognizer.recognize_conditions(self.pdf_path)
            self.assertEqual(requested, [1, 2, 3])
            cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self.assertEqual([item["task_num"] for item in cached["tasks"]], [1, 2])

            self.stdout.seek(0)
            self.stdout.truncate(0)
            client.models.generate_content.reset_mock()
            client.models.generate_content.side_effect = self.response
            result = recognizer.recognize_conditions(self.pdf_path)
        calls = client.models.generate_content.call_args_list
        numbers = [int(call.kwargs["contents"][0].rsplit(" ", 1)[1]) for call in calls]
        self.assertEqual(numbers, list(range(3, 20)))
        self.assertIn("Запрос OCR 1/17.", self.stdout.getvalue())
        self.assertIn("Запросов OCR: 17.", self.stdout.getvalue())
        for call in calls:
            image = call.kwargs["contents"][1].inline_data
            self.assertEqual(image.mime_type, "image/png")
            self.assertTrue(image.data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(len(result), 19)
        self.assertFalse(self.cache_path.with_suffix(".tmp").exists())

    def test_invalid_response_does_not_replace_progress(self):
        invalid = (
            "не JSON",
            "",
            json.dumps([{"task_num": 3, "condition": "Чужое задание"}]),
            json.dumps([{"task_num": 2, "condition": "[НЕРАЗБОРЧИВО]"}]),
        )
        for text in invalid:
            with self.subTest(text=text):
                self.save(self.state([1]))
                original = self.cache_path.read_bytes()
                client = MagicMock()
                client.models.generate_content.return_value = SimpleNamespace(text=text)
                with (
                    patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                    patch.object(recognizer, "render_task_images", side_effect=self.fake_images),
                    patch.object(recognizer.genai, "Client") as factory,
                ):
                    factory.return_value.__enter__.return_value = client
                    with self.assertRaises(ValueError):
                        recognizer.recognize_conditions(self.pdf_path)
                self.assertEqual(self.cache_path.read_bytes(), original)

    def test_different_cache_identity_starts_at_one_and_keeps_backup(self):
        changed = {
            "version": -1, "method": "old_pdf_method", "pdf_sha256": "other-pdf",
            "model": "other-model", "prompt": "old prompt", "dpi": 72,
        }
        for key, value in changed.items():
            with self.subTest(key=key):
                state = self.state(range(1, 20))
                state[key] = value
                self.save(state)
                original = self.cache_path.read_text(encoding="utf-8")
                client = MagicMock()
                client.models.generate_content.side_effect = TimeoutError("Сбой первого запроса")
                with (
                    patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                    patch.object(recognizer, "render_task_images", side_effect=self.fake_images),
                    patch.object(recognizer.genai, "Client") as factory,
                ):
                    factory.return_value.__enter__.return_value = client
                    with self.assertRaises(TimeoutError):
                        recognizer.recognize_conditions(self.pdf_path)
                self.assertEqual(
                    client.models.generate_content.call_args.kwargs["contents"][0],
                    "Номер задания: 1",
                )
                backups = (self.root / "variant_ocr").glob("conditions_before_*.json")
                self.assertTrue(any(path.read_text(encoding="utf-8") == original for path in backups))
                self.assertEqual(self.cache_path.read_text(encoding="utf-8"), original)

    def test_missing_key_does_not_modify_old_cache(self):
        self.save({"old": "старый кэш"})
        original = self.cache_path.read_bytes()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                recognizer.recognize_conditions(self.pdf_path)
        self.assertEqual(self.cache_path.read_bytes(), original)

    def test_validator_rejects_duplicate_numbers_and_unreadable_conditions(self):
        invalid = (
            [{"task_num": True, "condition": "Текст"}],
            [{"task_num": 1, "condition": " "}],
            [{"task_num": 1, "condition": "Символ \ufffd"}],
            [{"task_num": 1, "condition": "Символ \uf070"}],
            [{"task_num": 1, "condition": "[НЕРАЗБОРЧИВО]"}],
            [{"task_num": 1, "condition": "A"}, {"task_num": 1, "condition": "B"}],
        )
        for items in invalid:
            with self.subTest(items=items):
                with self.assertRaises(ValueError):
                    recognizer.validate_conditions(items, require_all=False)
        with self.assertRaises(ValueError):
            recognizer.validate_conditions([{"task_num": 1, "condition": "Текст"}])

    def make_numbered_pdf(self, count=19):
        with pymupdf.open() as document:
            for first in (1, 11):
                page = document.new_page(width=600, height=800)
                for number in range(first, min(first + 10, count + 1)):
                    offset = number - first
                    x = 25 if offset < 5 else 325
                    y = 50 + (offset % 5) * 130
                    page.insert_text((x, y), f"{number}. Task {number}")
            document.save(self.pdf_path)

    def test_two_column_pdf_produces_19_images_at_300_dpi(self):
        self.make_numbered_pdf()
        images = recognizer.render_task_images(self.pdf_path, self.root / "ocr")
        self.assertEqual(sorted(images), list(range(1, 20)))
        for number, path in images.items():
            self.assertEqual(path.name, f"task_{number:02d}.png")
            pixmap = pymupdf.Pixmap(path)
            self.assertEqual((pixmap.xres, pixmap.yres), (300, 300))

    def test_pdf_without_all_numbers_fails_before_recognition(self):
        self.make_numbered_pdf(count=18)
        with self.assertRaisesRegex(ValueError, r"\[19\]"):
            recognizer.render_task_images(self.pdf_path, self.root / "ocr")


if __name__ == "__main__":
    unittest.main()
