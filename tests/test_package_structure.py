"""Контракты переноса: CLI, точные промпты кэшей и безопасность ручной сверки."""

from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from alexlarin_parser import pipeline
from alexlarin_parser.classification import legacy, service
from alexlarin_parser.io import atomic_json
from alexlarin_parser.ocr import recognition
from alexlarin_parser.paths import PROJECT_ROOT
from alexlarin_parser.prompts import load_prompt


@pytest.mark.parametrize("resource,owner,attribute,digest", [
    ("ocr.txt", recognition, "PROMPT",
     "1f6100bb90af9d5a51ba2e58f6b6da39ecb006ba102500193dabc5b9542b71a0"),
    ("classification.txt", service, "VARIANT_PROMPT",
     "68ef9c1236e63e8f4fa1683e94e969933fdf3f3c7e991169f8f1f58f1d85ff21"),
    ("classification_numeric_legacy.txt", service, "NUMERIC_VARIANT_PROMPT",
     "6c643c1dd38d5ac010c57744c6f08e26cb734cc6abb07229eb4abae1432c9744"),
    ("classification_gemini_legacy.txt", legacy, "PROMPT",
     "262586deaf068be51b2d591070f4727e6c24fe3927f848fad53eeef0d9914a45"),
])
def test_extracted_prompt_keeps_exact_bytes_of_supported_cache_protocol(resource, owner, attribute, digest):
    """Пробелы/переносы опубликованного протокола не меняются при переносе файла."""
    text = load_prompt(resource)
    assert text == getattr(owner, attribute)
    assert sha256(text.encode("utf-8")).hexdigest() == digest


def test_package_entrypoint_and_legacy_script_show_same_options_from_supported_cwds(tmp_path):
    """Старый скрипт работает вне cwd проекта; новый пакет запускается из корня."""
    environment = {**os.environ, "DEEPSEEK_API_KEY": "", "GEMINI_API_KEY": ""}
    commands = [
        ([sys.executable, str(PROJECT_ROOT / "main.py"), "--help"], tmp_path),
        ([sys.executable, "-m", "alexlarin_parser", "--help"], PROJECT_ROOT),
    ]
    option_sets = []
    for command, cwd in commands:
        result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True,
                                text=True, timeout=15, check=False)
        assert result.returncode == 0, result.stderr
        option_sets.append({token for token in result.stdout.split() if token.startswith("--")})
    assert option_sets[0] == option_sets[1]
    assert {"--plan", "--local-only", "--max-variants", "--resume"} <= option_sets[0]


@pytest.mark.parametrize("module_path", [
    "exporting/service.py", "exporting/images.py", "exporting/workbook.py",
    "exporting/archive.py", "exporting/metadata.py", "formatting/latex.py",
    "formatting/math_text.py", "formatting/html.py",
])
def test_review_approval_is_invalidated_by_change_in_actual_packaged_formatter(tmp_path, monkeypatch, module_path):
    """Одобрение связано с реализацией, а не с неизменённым корневым фасадом."""
    context = SimpleNamespace(variant_key="ege_2013_trvar1", debug_dir=tmp_path / "debug")
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"unchanged source hash")
    sources = {"pdf_path": pdf, "metadata": {"year": 2013}}
    tasks = [{"task_num": 1, "condition": "$x^2$", "answer": "1"}]
    result = {"image_files": []}

    def apply():
        state = {"warnings": [], "missing_answers": []}
        pipeline._apply_content_review(context, sources, tasks, result, state)
        return state

    initial = apply()
    atomic_json(context.debug_dir / "content_review.json", {
        "fingerprint": initial["review_fingerprint"], "approved": True, "findings": [],
    })
    approved = apply()
    assert approved["content_reviewed"] is True and approved["status"] == "SUCCESS"

    actual_hash = pipeline.file_sha256
    changed_file = Path(pipeline.__file__).parent / module_path
    monkeypatch.setattr(pipeline, "file_sha256", lambda path:
                        "changed implementation" if Path(path) == changed_file else actual_hash(path))
    changed = apply()
    assert changed["review_fingerprint"] != initial["review_fingerprint"]
    assert changed["content_reviewed"] is False and changed["status"] == "NEEDS_REVIEW"
