"""Регрессии замечаний ревью: подставной HTTP и данные только в tmp_path."""
from pathlib import Path
import base64
import json
import socket
from unittest.mock import Mock

import pytest
import requests
import test_pipeline as fixtures
import test_provider as wire
from alexlarin_parser import pipeline, provider
from alexlarin_parser.classification import service
from alexlarin_parser.ocr import recognition
from alexlarin_parser import content_cache
from alexlarin_parser.io import atomic_json, fingerprint
from alexlarin_parser.model_response import ModelResponse, response_provenance


@pytest.fixture
def project(tmp_path, monkeypatch):
    return fixtures.project.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("stage", ["ocr", "classification"])
@pytest.mark.parametrize("nvidia_outcome", ["valid", "503", "timeout", "invalid_schema"])
def test_restored_openrouter_route_through_normal_pipeline(project, monkeypatch, stage, nvidia_outcome):
    """Настоящие этапы/валидаторы; HTTP подставной, сеть запрещена."""
    base, contexts, _ = project
    monkeypatch.setattr(provider.os, "environ", {})
    values = {"OCR_DPI": "72", "DEEPSEEK_API_KEY": "test-paid",
              "DEEPSEEK_CLASSIFICATION_THINKING": "enabled"}
    for prefix in ("GEMINI", "NVIDIA", "GROQ", "OPENROUTER"):
        values[f"{prefix}_API_KEY"] = f"test-{prefix}"
        for current in ("ocr", "classification"):
            values[f"{prefix}_{current.upper()}_MODEL"] = (
                "qwen/qwen3.5-122b-a10b" if prefix == "NVIDIA" else "test-model:free")
    # Gemini участвует только в проверяемом этапе, NVIDIA — в обоих.
    values[f"GEMINI_{'CLASSIFICATION' if stage == 'ocr' else 'OCR'}_MODEL"] = ""
    (base / ".env").write_text("\n".join(f"{key}={value}" for key, value in values.items()))
    reference = fixtures.FakeClient()
    calls = []
    nvidia_failed = False

    def post(url, **kwargs):
        nonlocal nvidia_failed
        payload = kwargs["json"]
        messages = payload["messages"]
        current = "ocr" if isinstance(messages[0]["content"], list) else "classification"
        calls.append((url, current, payload))
        assert url in {
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "https://integrate.api.nvidia.com/v1/chat/completions",
            "https://openrouter.ai/api/v1/chat/completions",
            "https://api.deepseek.com/chat/completions",
        }, "Отключённый провайдер получил HTTP-запрос"
        if "googleapis.com" in url:
            return wire.FakeResponse(status=503)
        if "openrouter.ai" in url:
            return wire.FakeResponse(status=503)
        if "nvidia.com" in url:
            assert payload["model"] == "qwen/qwen3.5-122b-a10b"
            assert "thinking" not in payload and "response_format" not in payload
            if current == "ocr":
                images = [part["image_url"]["url"] for part in messages[0]["content"]
                          if part["type"] == "image_url"]
                assert images and all(image.startswith("data:image/png;base64,") for image in images)
                assert all(base64.b64decode(image.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")
                           for image in images)
            else:
                tasks = json.loads(messages[1]["content"])["tasks"]
                assert len(tasks) == 2 and all(task["condition"] for task in tasks)
            if current == stage and not nvidia_failed and nvidia_outcome != "valid":
                nvidia_failed = True
                if nvidia_outcome == "503":
                    return wire.FakeResponse(status=503)
                if nvidia_outcome == "timeout":
                    raise requests.ReadTimeout("подставной тайм-аут")
                return fixtures.completion({"unexpected": True})
        return fixtures.completion(reference.request_json(messages, stage=current, debug_dir=contexts[0].debug_dir))

    session = Mock()
    session.post.side_effect = post
    monkeypatch.setattr(provider.requests, "Session", lambda: session)
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("Реальная сеть запрещена"))
    monkeypatch.setattr(socket.socket, "connect_ex", lambda *a: pytest.fail("Реальная сеть запрещена"))
    report = pipeline.run_batch(base, contexts[:1], local_only=True)
    state = report["variants"][0]
    assert state["error"] is None and state["result_current_attempt"] is True
    attempts = [attempt for attempt in state["attempts"] if attempt["stage"] == stage]
    expected = ["gemini", "nvidia-nim"] + ([] if nvidia_outcome == "valid" else ["openrouter", "deepseek-direct"])
    assert [attempt["provider"] for attempt in attempts[:len(expected)]] == expected
    assert [attempt["status"] for attempt in attempts[:len(expected)]] == (
        ["FAILED", "RECEIVED"] if nvidia_outcome == "valid" else ["FAILED", "FAILED", "FAILED", "RECEIVED"])
    if nvidia_outcome == "invalid_schema":
        assert "проверку схемы" in attempts[1]["error"]
    if nvidia_outcome == "valid":
        assert all(attempt["provider"] != "deepseek-direct" for attempt in state["attempts"])
    assert report["ocr_calls"] + report["classification_calls"] == len(calls) == len(state["attempts"])
    for _, current, payload in calls:
        if payload["model"] == "deepseek-flash":
            assert payload["thinking"] == {"type": "disabled" if current == "ocr" else "enabled"}
        else:
            assert "thinking" not in payload
    assert state[f"{stage}_provenance"]["provider"] == (
        "nvidia-nim" if nvidia_outcome == "valid" else
        "mixed" if stage == "ocr" and nvidia_outcome == "invalid_schema" else "deepseek-direct")
    session.close.assert_called_once()


@pytest.mark.parametrize("stage", ["ocr", "classification"])
def test_inactive_providers_cannot_be_injected_into_settings(tmp_path, stage):
    settings = provider.DeepSeekSettings(api_key="test-paid", free_providers=tuple(
        provider._FreeProvider(name, f"https://{name}.invalid/v1", "test-key", "vision", "text")
        for name in ("groq",)))
    session = wire.FakeSession(wire.FakeResponse())
    client = provider.DeepSeekClient(settings, session=session)
    client.request_json(wire.MESSAGES, stage=stage, debug_dir=tmp_path)
    assert [a["provider"] for a in client.attempts] == ["deepseek-direct"]
    assert "free_providers" not in settings.identity(stage)


def free_client(context):
    reference = fixtures.FakeClient()
    session = Mock()

    def post(url, **kwargs):
        messages = kwargs["json"]["messages"]
        stage = "ocr" if isinstance(messages[0]["content"], list) else "classification"
        return fixtures.completion(reference.request_json(messages, stage=stage, debug_dir=context.debug_dir))

    session.post.side_effect = post
    settings = provider.DeepSeekSettings(api_key="", dpi=72, free_providers=(
        provider._FreeProvider("gemini", "https://test.invalid/v1", "test-free", "vision", "text"),
    ))
    return provider.DeepSeekClient(settings, session=session)


def test_free_result_keeps_actual_provider_provenance(project):
    base, contexts, _ = project
    client = free_client(contexts[0])
    report = pipeline.run_batch(base, contexts[:1], client=client, local_only=True)
    state = report["variants"][0]
    assert {a["provider"] for a in state["attempts"]} == {"gemini"}
    assert state["ocr_provenance"]["provider"] == "gemini"
    assert state["classification_provenance"]["provider"] == "gemini"
    assert [page["provider"] for page in state["ocr_provenance"]["pages"]] == ["gemini", "gemini"]
    for path in (base / "input" / "cache" / "model_responses").rglob("*.json"):
        assert json.loads(path.read_text())["provenance"]["provider"] == "gemini"
    cached_client = free_client(contexts[0])
    cached_client.session.post.side_effect = AssertionError("Кэш не должен обращаться к модели")
    repeated = pipeline.run_batch(base, contexts[:1], client=cached_client, local_only=True)
    repeated_state = repeated["variants"][0]
    assert repeated_state["ocr_provenance"] == state["ocr_provenance"]
    assert repeated_state["classification_provenance"] == state["classification_provenance"]
    assert not cached_client.attempts


@pytest.mark.parametrize("stage", ["ocr", "classification"])
def test_optional_content_cache_failure_does_not_discard_valid_result(project, monkeypatch, caplog, stage):
    base, contexts, _ = project

    def fail_shared_cache(*args):
        raise PermissionError("общий кэш недоступен, локальный доступен")

    monkeypatch.setattr(recognition if stage == "ocr" else service, "write_response", fail_shared_cache)
    report = pipeline.run_batch(base, contexts[:1], client=fixtures.FakeClient(), local_only=True)
    assert (contexts[0].cache_dir / "classification.json").exists()
    assert report["variants"][0]["result_current_attempt"] is True
    assert "Общий кэш" in caplog.text


def test_deepseek_retry_after_blocks_next_paid_http(tmp_path):
    session = wire.FakeSession(wire.FakeResponse(status=429, headers={"Retry-After": "3600"}), wire.FakeResponse())
    client = provider.DeepSeekClient(provider.DeepSeekSettings(api_key="test-paid"), session=session)
    with pytest.raises(provider.ProviderError):
        client.request_json(wire.MESSAGES, stage="ocr", debug_dir=tmp_path)
    try:
        client.request_json(wire.MESSAGES, stage="classification", debug_dir=tmp_path)
    except provider.ProviderError:
        pass
    assert len(session.calls) == 1, "Второй POST отправлен до окончания Retry-After"


def test_interrupt_finishes_attempt_status(tmp_path):
    session = Mock()
    session.post.side_effect = KeyboardInterrupt()
    client = provider.DeepSeekClient(provider.DeepSeekSettings(api_key="test-paid"), session=session)
    with pytest.raises(KeyboardInterrupt):
        client.request_json(wire.MESSAGES, stage="ocr", debug_dir=tmp_path)
    assert client.counter.total == 1
    assert client.attempts[-1]["status"] == "INTERRUPTED"
    assert json.loads((tmp_path / "attempt_001_ocr.json").read_text())["status"] == "INTERRUPTED"


def test_retry_after_is_preserved_when_client_is_recreated(tmp_path):
    settings = provider.DeepSeekSettings(api_key="", free_providers=(
        provider._FreeProvider("gemini", "https://test.invalid/v1", "test-free", "vision", "text"),
    ), rate_limits=(provider.ProviderLimits("gemini", rpd=100),), rate_limit_path=tmp_path / "quotas.json")
    first = provider.DeepSeekClient(settings, session=wire.FakeSession(
        wire.FakeResponse(status=429, headers={"Retry-After": "3600"}),
    ))
    with pytest.raises(provider.ProviderError):
        first.request_json(wire.MESSAGES, stage="ocr", debug_dir=tmp_path / "first")
    second_session = wire.FakeSession(wire.FakeResponse())
    second = provider.DeepSeekClient(settings, session=second_session)
    try:
        second.request_json(wire.MESSAGES, stage="ocr", debug_dir=tmp_path / "second")
    except provider.ProviderError:
        pass
    assert not second_session.calls, "Новый клиент забыл серверный cooldown"


def test_resume_rejects_variant_key_not_matching_source_url(project):
    base, contexts, _ = project
    client = fixtures.FakeClient()
    client.failures[(contexts[0].variant_key, "ocr", 2)] = requests.Timeout("test")
    failed = pipeline.run_batch(base, contexts[:1], client=client, fresh_ai=True, local_only=True)
    manifest = Path(failed["manifest_path"])
    changed = json.loads(manifest.read_text())
    changed["variants"][0]["variant_key"] = "ege_2013_trvar999"
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        resumed = pipeline.selected_contexts(base, resume=manifest, local_only=True)
        pipeline.run_batch(base, resumed, client=fixtures.FakeClient(), resume_from=manifest, local_only=True)


@pytest.mark.parametrize("invalid_stage", ["ocr", "classification"])
def test_schema_invalid_free_answer_can_fall_back(project, invalid_stage):
    base, contexts, _ = project
    client = free_client(contexts[0])
    good_post = client.session.post.side_effect
    calls = []

    def post(url, **kwargs):
        messages = kwargs["json"]["messages"]
        stage = "ocr" if isinstance(messages[0]["content"], list) else "classification"
        first_invalid = stage == invalid_stage and not any(s == invalid_stage for _, s in calls)
        calls.append((url, stage))
        if first_invalid:
            return fixtures.completion({"unexpected": True})
        return good_post(url, **kwargs)

    client.session.post.side_effect = post
    client.settings = provider.DeepSeekSettings(api_key="", dpi=72, free_providers=(
        provider._FreeProvider("gemini", "https://test.invalid/v1", "test-free", "vision", "text"),
        provider._FreeProvider("nvidia-nim", "https://next.invalid/v1", "test-next", "vision", "text"),
    ))
    report = pipeline.run_batch(base, contexts[:1], client=client, local_only=True)
    assert report["variants"][0]["error"] is None
    attempts = [a for a in client.attempts if a["stage"] == invalid_stage]
    assert [(a["provider"], a["status"]) for a in attempts[:2]] == [("gemini", "FAILED"), ("nvidia-nim", "RECEIVED")]


def test_content_cache_provenance_survives_moving_to_another_variant(project):
    base, contexts, sources = project
    first_client = free_client(contexts[0])
    original = recognition.recognize_variant(contexts[0], sources[contexts[0].variant_key], first_client)
    current = free_client(contexts[1])
    current.session.post.side_effect = AssertionError("Тот же вход уже сохранён в общем кэше")
    copied = recognition.recognize_variant(contexts[1], sources[contexts[0].variant_key], current)
    assert copied["provenance"] == original["provenance"]
    assert copied["provenance"]["provider"] == "gemini" and not current.attempts
    assert json.loads((contexts[1].cache_dir / "ocr" / "page_001.json").read_text())["provenance"]["provider"] == "gemini"


def test_old_chain_cache_does_not_invent_provider(project):
    base, contexts, _ = project
    pipeline.run_batch(base, contexts[:1], client=free_client(contexts[0]))
    paths = [contexts[0].cache_dir / "classification.json", *(contexts[0].cache_dir / "ocr").glob("*.json")]
    for path in paths:
        record = json.loads(path.read_text())
        record.pop("provenance")
        atomic_json(path, record)
    before = {path: path.read_bytes() for path in paths}
    client = free_client(contexts[0])
    client.session.post.side_effect = AssertionError("Исторические данные должны читаться без API")
    result = pipeline.run_batch(base, contexts[:1], client=client)
    state = result["variants"][0]
    assert state["ocr_provenance"]["provider"] == state["classification_provenance"]["provider"] == "unknown"
    assert all(path.read_bytes() == data for path, data in before.items())


def test_shared_cache_checks_provenance_integrity_and_accepts_old_records(tmp_path):
    path, identity = tmp_path / "content.json", {"test": "same-request"}
    data = ModelResponse({"tasks": []}, {"provider": "gemini", "model": "vision", "base_url": "https://test.invalid"})
    content_cache.write_response(path, identity, data)
    loaded = content_cache.read_response(path, identity, lambda data: None)
    assert loaded == data and loaded.provenance == data.provenance
    record = json.loads(path.read_text())
    record["provenance"]["provider"] = "another-provider"
    atomic_json(path, record)
    assert content_cache.read_response(path, identity, lambda data: None) is None
    atomic_json(path, {"identity": identity, "response": data, "response_sha256": fingerprint(data)})
    loaded = content_cache.read_response(path, identity, lambda data: None)
    assert loaded == data
    assert response_provenance(loaded, {"free_providers": [{}]})["provider"] == "unknown"


@pytest.mark.parametrize("name,prefix", [
    ("gemini", "GEMINI"), ("nvidia-nim", "NVIDIA"), ("openrouter", "OPENROUTER"),
    ("deepseek-direct", "DEEPSEEK"),
])
def test_server_cooldown_survives_restart_even_without_local_quotas(tmp_path, monkeypatch, name, prefix):
    monkeypatch.setattr(provider.os, "environ", {f"{prefix}_API_KEY": "test-key", f"{prefix}_OCR_MODEL": "test:free"})
    settings = provider.DeepSeekSettings.from_env(tmp_path)
    assert not settings.rate_limits
    now = [100000.0]
    first = provider.DeepSeekClient(settings, session=wire.FakeSession(
        wire.FakeResponse(status=429, headers={"Retry-After": "37"}),
    ), rate_limiter=provider.RequestRateLimiter(settings.rate_limit_path, clock=lambda: now[0]))
    with pytest.raises(provider.ProviderError):
        first.request_json(wire.MESSAGES, stage="ocr", debug_dir=tmp_path / "first")
    now[0] += 36
    session = wire.FakeSession(wire.FakeResponse())
    second = provider.DeepSeekClient(settings, session=session,
                                     rate_limiter=provider.RequestRateLimiter(settings.rate_limit_path, clock=lambda: now[0]))
    with pytest.raises(provider.ProviderError):
        second.request_json(wire.MESSAGES, stage="ocr", debug_dir=tmp_path / "second")
    assert not session.calls and not second.attempts
    now[0] += 1
    second.request_json(wire.MESSAGES, stage="ocr", debug_dir=tmp_path / "second")
    assert len(session.calls) == second.counter.total == 1


@pytest.mark.parametrize("corruption", ["duplicate", "wrong_cache", "other_context"])
def test_resume_validation_happens_before_any_client_creation(project, monkeypatch, corruption):
    base, contexts, _ = project
    failed = pipeline.run_batch(base, contexts[:1], client=fixtures.FakeClient())
    manifest = Path(failed["manifest_path"])
    data = json.loads(manifest.read_text())
    if corruption == "duplicate":
        data["variants"].append(data["variants"][0])
    elif corruption == "wrong_cache":
        data["variants"][0]["cache_dir"] = str(contexts[1].cache_dir)
    atomic_json(manifest, data)
    monkeypatch.setattr(pipeline, "DeepSeekClient", lambda *a, **k: pytest.fail("Неверный resume не создаёт клиент"))
    with pytest.raises(ValueError):
        pipeline.run_batch(base, contexts[1:] if corruption == "other_context" else contexts[:1], resume_from=manifest)
