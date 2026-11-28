"""Причины отказов/пропусков цепочки: только подставной HTTP и tmp_path."""

import json
import socket
from unittest.mock import Mock

import pytest
import requests

import test_pipeline as fixtures
import test_provider as wire
from alexlarin_parser import pipeline, provider


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(provider.os, "environ", {})
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("Реальная сеть запрещена"))
    monkeypatch.setattr(socket.socket, "connect_ex", lambda *a: pytest.fail("Реальная сеть запрещена"))


@pytest.fixture
def project(tmp_path, monkeypatch):
    return fixtures.project.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("stage", ["ocr", "classification"])
def test_pipeline_error_keeps_http_failure_and_each_configuration_skip(project, monkeypatch, stage):
    base, contexts, _ = project
    other_stage = "classification" if stage == "ocr" else "ocr"
    (base / ".env").write_text(
        "OCR_DPI=72\nDEEPSEEK_API_KEY=\nGEMINI_API_KEY=test-gemini\n"
        f"GEMINI_{stage.upper()}_MODEL=test-model\n"
        f"NVIDIA_API_KEY=test-nvidia\nNVIDIA_{other_stage.upper()}_MODEL=test-model\n")
    reference = fixtures.FakeClient()
    session = Mock()

    def post(url, **kwargs):
        if "googleapis.com" in url:
            return wire.FakeResponse(status=503)
        assert "nvidia.com" in url and stage == "classification"
        return fixtures.completion(reference.request_json(
            kwargs["json"]["messages"], stage="ocr", debug_dir=contexts[0].debug_dir))

    session.post.side_effect = post
    monkeypatch.setattr(provider.requests, "Session", lambda: session)
    report = pipeline.run_batch(base, contexts[:1], local_only=True)
    state = report["variants"][0]
    assert state["status"] == "FAILED"
    assert all(fragment in state["error"] for fragment in (
        stage, "gemini", "503", "nvidia-nim", f"NVIDIA_{stage.upper()}_MODEL",
        "openrouter", "OPENROUTER_API_KEY", "deepseek-direct", "DEEPSEEK_API_KEY"))
    diagnostics = [d for d in state["provider_diagnostics"] if d["stage"] == stage]
    assert [(d["provider"], d["status"], d["kind"]) for d in diagnostics] == [
        ("gemini", "FAILED", "http"),
        ("nvidia-nim", "SKIPPED", "configuration"),
        ("groq", "SKIPPED", "configuration"),
        ("openrouter", "SKIPPED", "configuration"),
        ("deepseek-direct", "SKIPPED", "configuration"),
    ]
    assert diagnostics[0]["http_status"] == 503 and diagnostics[0]["attempt"] >= 1
    assert all("attempt" not in d for d in diagnostics[1:])
    assert report["ocr_calls"] + report["classification_calls"] == session.post.call_count == (
        1 if stage == "ocr" else 3)
    saved = json.loads((base / "reports" / report["run_id"] / f"{contexts[0].variant_key}.json").read_text())
    assert saved["provider_diagnostics"] == state["provider_diagnostics"]


def all_provider_settings(tmp_path, monkeypatch, **extra):
    for prefix in ("GEMINI", "NVIDIA", "OPENROUTER", "DEEPSEEK"):
        monkeypatch.setenv(f"{prefix}_API_KEY", f"test-{prefix}")
        for stage in ("OCR", "CLASSIFICATION"):
            monkeypatch.setenv(f"{prefix}_{stage}_MODEL", "test-model:free")
    for name, value in extra.items():
        monkeypatch.setenv(name, value)
    return provider.DeepSeekSettings.from_env(tmp_path)


@pytest.mark.parametrize("stage", ["ocr", "classification"])
def test_final_error_collects_timeout_http_validation_and_paid_block(tmp_path, monkeypatch, stage):
    settings = all_provider_settings(tmp_path, monkeypatch)
    session = wire.FakeSession(wire.FakeResponse(status=503), requests.ReadTimeout("private-request-details"),
                               wire.FakeResponse(), wire.FakeResponse(status=401))
    client = provider.DeepSeekClient(settings, session=session)

    def validate(data):
        raise ValueError("нет обязательных полей " + "test-OPENROUTER")

    with pytest.raises(provider.PaidCallsBlocked) as failure:
        client.request_json(wire.MESSAGES, stage=stage, debug_dir=tmp_path / "debug", validate=validate)
    message = str(failure.value)
    assert all(fragment in message for fragment in (stage, "503", "ReadTimeout", "проверку схемы", "401"))
    assert "private-request-details" not in message and "test-OPENROUTER" not in message
    assert [d["kind"] for d in client.provider_diagnostics] == ["http", "timeout", "configuration", "validation", "http"]
    assert all(d["status"] == "FAILED" for d in client.provider_diagnostics if d["provider"] != "groq")
    assert len(session.calls) == client.counter.total == len(client.attempts) == 4


@pytest.mark.parametrize("blocker", ["quota", "cooldown", "disabled"])
def test_skips_explain_reason_without_counting_http_or_reusing_previous_stage(tmp_path, monkeypatch, blocker):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_OCR_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_CLASSIFICATION_MODEL", "test-model")
    if blocker == "quota":
        monkeypatch.setenv("GEMINI_RPD", "1")
    settings = provider.DeepSeekSettings.from_env(tmp_path)
    outcome = wire.FakeResponse(status={"quota": 200, "cooldown": 429, "disabled": 403}[blocker],
                                headers={"Retry-After": "37"})
    first = provider.DeepSeekClient(settings, session=wire.FakeSession(outcome))
    if blocker == "quota":
        first.request_json(wire.MESSAGES, stage="ocr", debug_dir=tmp_path / "first")
    else:
        with pytest.raises(provider.ProviderError):
            first.request_json(wire.MESSAGES, stage="ocr", debug_dir=tmp_path / "first")
    # Квота/cooldown должны объясняться и после пересоздания клиента.
    client = first if blocker == "disabled" else provider.DeepSeekClient(settings, session=wire.FakeSession())
    offset = len(client.provider_diagnostics)
    with pytest.raises(provider.ProviderError) as failure:
        client.request_json(wire.MESSAGES, stage="classification", debug_dir=tmp_path / "second")
    events = client.provider_diagnostics[offset:]
    assert len(events) == 5 and all(event["stage"] == "classification" for event in events)
    assert events[0]["kind"] == blocker and events[0]["status"] == "SKIPPED"
    assert ("RPD" if blocker == "quota" else "cooldown" if blocker == "cooldown" else "403") in events[0]["reason"]
    assert "classification" in str(failure.value) and "ocr" not in str(failure.value)
    assert all("attempt" not in event for event in events)
    assert client.counter.total == (1 if blocker == "disabled" else 0)
    assert not list((tmp_path / "second").glob("attempt_*.json"))


def test_success_keeps_preceding_skips_and_marks_unneeded_providers(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("NVIDIA_OCR_MODEL", "test-model")
    client = provider.DeepSeekClient(provider.DeepSeekSettings.from_env(tmp_path),
                                     session=wire.FakeSession(wire.FakeResponse()))
    client.request_json(wire.MESSAGES, stage="ocr", debug_dir=tmp_path / "debug")
    assert [(d["provider"], d["status"], d["kind"]) for d in client.provider_diagnostics] == [
        ("gemini", "SKIPPED", "configuration"), ("nvidia-nim", "RECEIVED", "http"),
        ("groq", "SKIPPED", "not_needed"), ("openrouter", "SKIPPED", "not_needed"), ("deepseek-direct", "SKIPPED", "not_needed"),
    ]
    assert client.counter.total == 1


def test_restored_openrouter_completes_both_pipeline_stages_without_paid_key(project, monkeypatch):
    base, contexts, _ = project
    settings = all_provider_settings(base, monkeypatch, DEEPSEEK_API_KEY="", OCR_DPI="72")
    # Проверяется создание клиента обычным run_batch, а не внедрение готового клиента.
    reference = fixtures.FakeClient()
    session = Mock()

    def post(url, **kwargs):
        if "googleapis.com" in url or "nvidia.com" in url:
            return wire.FakeResponse(status=503)
        assert url == "https://openrouter.ai/api/v1/chat/completions"
        payload = kwargs["json"]
        assert payload["model"].endswith(":free") and "thinking" not in payload
        assert payload["response_format"] == {"type": "json_object"}
        messages = payload["messages"]
        stage = "ocr" if isinstance(messages[0]["content"], list) else "classification"
        return fixtures.completion(reference.request_json(messages, stage=stage, debug_dir=contexts[0].debug_dir))

    session.post.side_effect = post
    monkeypatch.setattr(provider.requests, "Session", lambda: session)
    report = pipeline.run_batch(base, contexts[:1], local_only=True)
    state = report["variants"][0]
    assert state["error"] is None and state["result_current_attempt"] is True
    assert state["ocr_provenance"]["provider"] == state["classification_provenance"]["provider"] == "openrouter"
    assert [a["provider"] for a in state["attempts"]] == ["gemini", "nvidia-nim"] + ["openrouter"] * 3
    assert session.post.call_count == report["ocr_calls"] + report["classification_calls"] == 5
    assert settings.api_key == ""
