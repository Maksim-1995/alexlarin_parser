"""Постраничный OCR и его кэши проверяются без сетевых запросов."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pymupdf
import pytest

import recognizer
from ai_requests import AIRequestCounter


PAGE_NUMBERS = (list(range(1, 5)), list(range(5, 13)), list(range(13, 20)))


def make_pdf(path, page_numbers=PAGE_NUMBERS, changed_page=None):
    with pymupdf.open() as document:
        for page_num, numbers in enumerate(page_numbers, start=1):
            page = document.new_page(width=360, height=480)
            for index, number in enumerate(numbers):
                half = (len(numbers) + 1) // 2
                x = 20 if index < half else 200
                y = 45 + (index % half) * 48
                page.insert_text((x, y), f"{number}. Task {number}")
            if page_num == changed_page:
                page.insert_text((20, 400), "Additional information")
        document.save(path)


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "variant.pdf"
    make_pdf(path)
    return path


def full_cache_path(pdf):
    return pdf.with_name(f"{pdf.stem}_conditions_gemini.json")


def page_cache_path(pdf, page_num):
    return pdf.parent / f"{pdf.stem}_ocr" / f"page_{page_num:02d}.json"


def legacy_state(pdf, numbers=range(1, 20)):
    return {
        **recognizer.legacy_cache_identity(pdf),
        "tasks": [{"task_num": n, "condition": f"Условие {n}"} for n in numbers],
    }


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def response_for(page_num, page_numbers=PAGE_NUMBERS):
    return {
        "page_num": page_num,
        "tasks": [
            {"task_num": n, "condition": rf"А) Найдите $x_{n}$. Б) Сравните $\sqrt{{{n}}}$."}
            for n in page_numbers[page_num - 1]
        ],
    }


def number_from_request(request):
    return int(request["contents"][0].rsplit(": ", 1)[1])


def install_client(monkeypatch, side_effect=None):
    client = MagicMock()
    if side_effect is None:
        side_effect = lambda **request: SimpleNamespace(
            text=json.dumps(response_for(number_from_request(request)), ensure_ascii=False)
        )
    client.models.generate_content.side_effect = side_effect
    factory = MagicMock()
    factory.return_value.__enter__.return_value = client
    monkeypatch.setattr(recognizer.genai, "Client", factory)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return client, factory


def forbid_client(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("При пригодном OCR-кэше SDK и API не нужны")
    monkeypatch.setattr(recognizer.genai, "Client", forbidden)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_three_full_pages_need_exactly_three_requests(pdf, monkeypatch, capsys):
    client, factory = install_client(monkeypatch)
    result = recognizer.recognize_conditions(pdf)
    assert sorted(result) == list(range(1, 20))
    assert client.models.generate_content.call_count == 3
    assert factory.call_count == 1
    options = factory.call_args.kwargs["http_options"]
    assert options.timeout == 120_000
    assert options.retry_options.attempts == 1
    for page_num, call in enumerate(client.models.generate_content.call_args_list, start=1):
        request = call.kwargs
        assert request["model"] == recognizer.MODEL
        assert request["contents"][0] == f"Номер страницы: {page_num}"
        assert request["contents"][2] == recognizer.PAGE_PROMPT
        image = request["contents"][1].inline_data
        assert image.mime_type == "image/png"
        pixmap = pymupdf.Pixmap(image.data)
        # Полная страница 360x480 пунктов при 300 dpi, не область задания.
        assert (pixmap.width, pixmap.height) == (1500, 2000)
        assert request["config"].max_output_tokens == 16384
        assert request["config"].response_json_schema == recognizer.SCHEMA
        assert page_cache_path(pdf, page_num).is_file()
    data = json.loads(full_cache_path(pdf).read_text(encoding="utf-8"))
    assert data["version"] == recognizer.AGGREGATE_CACHE_VERSION
    assert len(data["tasks"]) == 19
    assert "Б)" in result[19] and r"\sqrt" in result[19]
    assert "Запрос OCR 1/3." in capsys.readouterr().out


def test_complete_legacy_cache_survives_new_page_method_without_api_or_render(pdf, monkeypatch, capsys):
    state = legacy_state(pdf)
    assert state["version"] == 2
    assert state["method"] == "gemini_image_ocr"
    write_json(full_cache_path(pdf), state)
    before = full_cache_path(pdf).read_bytes()
    forbid_client(monkeypatch)
    monkeypatch.setattr(recognizer, "render_page_images", lambda *args: pytest.fail("Рендер не нужен"))
    monkeypatch.setattr(recognizer, "PAGE_PROMPT", recognizer.PAGE_PROMPT + " Уточнение нового метода.")
    result = recognizer.recognize_conditions(pdf)
    assert sorted(result) == list(range(1, 20))
    assert full_cache_path(pdf).read_bytes() == before
    assert "Запросов OCR: 0." in capsys.readouterr().out


def test_current_complete_cache_needs_no_sdk_or_images(pdf, monkeypatch):
    client, _ = install_client(monkeypatch)
    expected = recognizer.recognize_conditions(pdf)
    assert client.models.generate_content.call_count == 3
    before = full_cache_path(pdf).read_bytes()
    forbid_client(monkeypatch)
    monkeypatch.setattr(recognizer, "render_page_images", lambda *args: pytest.fail("Рендер не нужен"))
    assert recognizer.recognize_conditions(pdf) == expected
    assert full_cache_path(pdf).read_bytes() == before


def test_separate_cache_root_ignores_and_preserves_working_full_and_page_caches(
    pdf, tmp_path, monkeypatch,
):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    original_pdf = source_dir / pdf.name
    pdf.replace(original_pdf)
    install_client(monkeypatch)
    expected = recognizer.recognize_conditions(original_pdf)
    original_files = {
        path: path.read_bytes() for path in source_dir.rglob("*") if path.is_file()
    }
    cache_dir = tmp_path / "isolated-ai-cache"
    client, _ = install_client(monkeypatch)
    counter = AIRequestCounter()
    result = recognizer.recognize_conditions(
        original_pdf, cache_dir=cache_dir, request_counter=counter,
    )
    assert result == expected
    assert client.models.generate_content.call_count == counter.ocr == 3
    assert counter.classification == 0 and counter.total == 3
    assert all(path.read_bytes() == data for path, data in original_files.items())
    assert {path for path in source_dir.rglob("*") if path.is_file()} == set(original_files)
    aggregate = json.loads(
        (cache_dir / "variant_conditions_gemini.json").read_text(encoding="utf-8")
    )
    assert aggregate["page_caches"] == [
        f"variant_ocr/page_{number:02d}.json" for number in range(1, 4)
    ]
    assert all((cache_dir / name).is_file() for name in aggregate["page_caches"])

    forbid_client(monkeypatch)
    cached_counter = AIRequestCounter()
    assert recognizer.recognize_conditions(
        original_pdf, cache_dir=cache_dir, request_counter=cached_counter,
    ) == expected
    assert cached_counter.total == 0


def test_page_caches_rebuild_missing_aggregate_without_key_or_api(pdf, monkeypatch, capsys):
    install_client(monkeypatch)
    expected = recognizer.recognize_conditions(pdf)
    full_cache_path(pdf).unlink()
    forbid_client(monkeypatch)
    assert recognizer.recognize_conditions(pdf) == expected
    assert full_cache_path(pdf).is_file()
    assert "Запросов OCR: 0." in capsys.readouterr().out


def test_api_failure_keeps_checked_page_and_resume_requests_only_missing_pages(pdf, monkeypatch, capsys):
    requested = []

    def fail_second(**request):
        page_num = number_from_request(request)
        requested.append(page_num)
        if page_num == 2:
            raise TimeoutError("Тестовая ошибка API")
        return SimpleNamespace(text=json.dumps(response_for(page_num)))

    client, _ = install_client(monkeypatch, fail_second)
    counter = AIRequestCounter()
    with pytest.raises(RuntimeError, match="Страница 2.*Автоматических повторов нет"):
        recognizer.recognize_conditions(pdf, request_counter=counter)
    assert requested == [1, 2]
    assert client.models.generate_content.call_count == 2
    assert counter.ocr == counter.total == 2
    first = page_cache_path(pdf, 1).read_bytes()
    assert not page_cache_path(pdf, 2).exists()
    assert not full_cache_path(pdf).exists()
    client, _ = install_client(monkeypatch)
    result = recognizer.recognize_conditions(pdf)
    assert sorted(result) == list(range(1, 20))
    assert [number_from_request(call.kwargs) for call in client.models.generate_content.call_args_list] == [2, 3]
    assert page_cache_path(pdf, 1).read_bytes() == first
    assert "Запросов OCR: 2." in capsys.readouterr().out


@pytest.mark.parametrize("invalid", [
    "не JSON", "", [], {"page_num": True, "tasks": []},
    {"page_num": 2, "tasks": []}, {"page_num": 1, "tasks": []},
    {"page_num": 1, "tasks": [{"task_num": 1, "condition": " "}]},
    {"page_num": 1, "tasks": [{"task_num": 1, "condition": "A"}, {"task_num": 1, "condition": "B"}]},
])
def test_invalid_page_response_stops_without_automatic_repair(pdf, monkeypatch, invalid):
    text = invalid if isinstance(invalid, str) else json.dumps(invalid)
    client, _ = install_client(monkeypatch, lambda **request: SimpleNamespace(text=text))
    counter = AIRequestCounter()
    with pytest.raises(ValueError, match="Страница 1.*Дополнительный запрос не выполняется"):
        recognizer.recognize_conditions(pdf, request_counter=counter)
    assert client.models.generate_content.call_count == 1
    assert counter.ocr == counter.total == 1
    assert not page_cache_path(pdf, 1).exists()
    assert not full_cache_path(pdf).exists()


def test_token_limit_is_rejected_even_if_returned_json_looks_complete(pdf, monkeypatch):
    client, _ = install_client(monkeypatch, lambda **request: SimpleNamespace(
        text=json.dumps(response_for(1)),
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
    ))
    with pytest.raises(ValueError, match="MAX_TOKENS"):
        recognizer.recognize_conditions(pdf)
    assert client.models.generate_content.call_count == 1
    assert not page_cache_path(pdf, 1).exists()


def test_invalid_second_page_keeps_checked_page_and_previous_aggregate(pdf, monkeypatch):
    write_json(full_cache_path(pdf), legacy_state(pdf, range(1, 17)))
    before = full_cache_path(pdf).read_bytes()

    def invalid_second(**request):
        page_num = number_from_request(request)
        data = response_for(page_num)
        if page_num == 2:
            data["tasks"][0]["task_num"] = 1
        return SimpleNamespace(text=json.dumps(data))

    client, _ = install_client(monkeypatch, invalid_second)
    with pytest.raises(ValueError, match="Страница 2"):
        recognizer.recognize_conditions(pdf)
    assert client.models.generate_content.call_count == 2
    assert page_cache_path(pdf, 1).is_file()
    assert not page_cache_path(pdf, 2).exists()
    assert full_cache_path(pdf).read_bytes() == before


@pytest.mark.parametrize("setting", ["model", "page_prompt", "dpi", "max_tokens", "schema"])
def test_page_identity_settings_invalidate_cache(pdf, monkeypatch, setting):
    install_client(monkeypatch)
    recognizer.recognize_conditions(pdf)
    if setting == "model":
        monkeypatch.setattr(recognizer, "MODEL", "test-new-model")
    elif setting == "page_prompt":
        monkeypatch.setattr(recognizer, "PAGE_PROMPT", recognizer.PAGE_PROMPT + " Уточнение.")
    elif setting == "dpi":
        monkeypatch.setattr(recognizer, "DPI", 150)
    elif setting == "max_tokens":
        monkeypatch.setattr(recognizer, "MAX_OUTPUT_TOKENS", recognizer.MAX_OUTPUT_TOKENS + 1)
    else:
        schema = copy.deepcopy(recognizer.SCHEMA)
        schema["description"] = "Новый контракт"
        monkeypatch.setattr(recognizer, "SCHEMA", schema)
    client, _ = install_client(monkeypatch)
    assert len(recognizer.recognize_conditions(pdf)) == 19
    assert client.models.generate_content.call_count == 3


def test_changing_one_pdf_page_reuses_other_page_caches(pdf, monkeypatch):
    install_client(monkeypatch)
    recognizer.recognize_conditions(pdf)
    old_first = page_cache_path(pdf, 1).read_bytes()
    old_third = page_cache_path(pdf, 3).read_bytes()
    changed = pdf.with_name("changed.pdf")
    make_pdf(changed, changed_page=2)
    changed.replace(pdf)
    client, _ = install_client(monkeypatch)
    recognizer.recognize_conditions(pdf)
    assert [number_from_request(call.kwargs) for call in client.models.generate_content.call_args_list] == [2]
    assert page_cache_path(pdf, 1).read_bytes() == old_first
    assert page_cache_path(pdf, 3).read_bytes() == old_third


def test_corrupt_page_cache_is_replaced_only_for_that_page(pdf, monkeypatch):
    install_client(monkeypatch)
    recognizer.recognize_conditions(pdf)
    full_cache_path(pdf).unlink()
    page_cache_path(pdf, 2).write_text("не JSON", encoding="utf-8")
    client, _ = install_client(monkeypatch)
    recognizer.recognize_conditions(pdf)
    assert [number_from_request(call.kwargs) for call in client.models.generate_content.call_args_list] == [2]


def test_missing_key_preserves_old_partial_legacy_cache(pdf, monkeypatch):
    write_json(full_cache_path(pdf), legacy_state(pdf, range(1, 17)))
    before = full_cache_path(pdf).read_bytes()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = MagicMock()
    monkeypatch.setattr(recognizer.genai, "Client", client)
    counter = AIRequestCounter()
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        recognizer.recognize_conditions(pdf, request_counter=counter)
    client.assert_not_called()
    assert counter.total == 0
    assert full_cache_path(pdf).read_bytes() == before


@pytest.mark.parametrize("failure", ["client_initialization", "image_preparation"])
def test_counter_stays_zero_when_ocr_request_cannot_be_prepared(pdf, monkeypatch, failure):
    client, _ = install_client(monkeypatch)
    counter = AIRequestCounter()

    def fail(**kwargs):
        raise ValueError("Ошибка подготовки запроса")

    if failure == "client_initialization":
        monkeypatch.setattr(recognizer.genai, "Client", fail)
    else:
        monkeypatch.setattr(recognizer.types.Part, "from_bytes", fail)
    with pytest.raises((ValueError, RuntimeError), match="Ошибка подготовки запроса"):
        recognizer.recognize_conditions(pdf, request_counter=counter)
    client.models.generate_content.assert_not_called()
    assert counter.total == 0


def test_atomic_page_cache_failure_preserves_old_file_and_cleans_pending(pdf, monkeypatch):
    old = {"old": "Неактуальная, но сохранённая версия"}
    path = page_cache_path(pdf, 1)
    path.parent.mkdir()
    write_json(path, old)
    before = path.read_bytes()
    original_replace = Path.replace

    def failed_replace(pending, destination):
        if Path(destination) == path:
            assert path.read_bytes() == before
            raise OSError("Тестовый сбой записи")
        return original_replace(pending, destination)

    monkeypatch.setattr(Path, "replace", failed_replace)
    client, _ = install_client(monkeypatch)
    with pytest.raises(OSError, match="сбой записи"):
        recognizer.recognize_conditions(pdf)
    assert client.models.generate_content.call_count == 1
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))


def test_page_sets_are_read_from_pdf_instead_of_hardcoded_ranges(pdf, monkeypatch):
    alternate = (list(range(1, 8)), list(range(8, 20)))
    changed = pdf.with_name("alternate.pdf")
    make_pdf(changed, page_numbers=alternate)
    changed.replace(pdf)
    client, _ = install_client(monkeypatch, lambda **request: SimpleNamespace(
        text=json.dumps(response_for(number_from_request(request), alternate))
    ))
    assert len(recognizer.recognize_conditions(pdf)) == 19
    assert client.models.generate_content.call_count == 2


def test_page_without_tasks_still_gets_one_request(pdf, monkeypatch):
    pages = ([], *PAGE_NUMBERS)
    changed = pdf.with_name("with_title.pdf")
    make_pdf(changed, page_numbers=pages)
    changed.replace(pdf)
    client, _ = install_client(monkeypatch, lambda **request: SimpleNamespace(
        text=json.dumps(response_for(number_from_request(request), pages))
    ))
    assert len(recognizer.recognize_conditions(pdf)) == 19
    assert client.models.generate_content.call_count == 4
    title = json.loads(page_cache_path(pdf, 1).read_text(encoding="utf-8"))
    assert title["response"] == {"page_num": 1, "tasks": []}


@pytest.mark.parametrize("numbers", [
    (list(range(1, 10)), list(range(10, 19))),
    ([1, 1, *range(2, 10)], list(range(10, 20))),
])
def test_incomplete_or_duplicate_pdf_numbers_fail_before_api(pdf, monkeypatch, numbers):
    changed = pdf.with_name("incomplete.pdf")
    make_pdf(changed, page_numbers=numbers)
    changed.replace(pdf)
    client, factory = install_client(monkeypatch)
    with pytest.raises(ValueError, match="1–19 без повторов"):
        recognizer.recognize_conditions(pdf)
    client.models.generate_content.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize("items", [
    [{"task_num": True, "condition": "Текст"}],
    [{"task_num": 1, "condition": " "}],
    [{"task_num": 1, "condition": "Символ \ufffd"}],
    [{"task_num": 1, "condition": "Символ \uf070"}],
    [{"task_num": 1, "condition": "[НЕРАЗБОРЧИВО]"}],
    [{"task_num": 1, "condition": "A"}, {"task_num": 1, "condition": "B"}],
    [{"task_num": 1, "condition": "Текст", "answer": "Не требуется"}],
])
def test_validator_rejects_bad_conditions(items):
    with pytest.raises(ValueError):
        recognizer.validate_conditions(items, require_all=False)
