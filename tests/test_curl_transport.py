"""Groq HTTP, fallback и тайм-ауты: curl подменён, внешней сети нет."""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from alexlarin_parser import curl_transport, provider


def fake_curl(monkeypatch, status=200, returncode=0, response_factory=None):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        options = {}
        headers = []
        for line in kwargs["input"].splitlines():
            key, separator, value = line.partition(" = ")
            if separator:
                parsed = json.loads(value) if value.startswith('"') else value
                if key == "header":
                    headers.append(parsed)
                else:
                    options[key] = parsed
        calls[-1] += (options, headers)
        body = (response_factory(json.loads(options["data-binary"])) if response_factory else
                {"choices": [{"finish_reason": "stop", "message": {"content": '{"tasks": []}'}}]})
        Path(options["output"]).write_text(json.dumps(body))
        Path(options["dump-header"]).write_text(
            f"HTTP/1.1 200 Connection established\r\nX-Proxy: hidden\r\n\r\n"
            f"HTTP/2 {status}\r\nRetry-After: 37\r\nContent-Type: application/json\r\n\r\n")
        return SimpleNamespace(returncode=returncode, stdout=str(status), stderr="private")

    monkeypatch.setattr(curl_transport.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("stage", ["ocr", "classification"])
def test_default_groq_profile_uses_curl_and_records_provenance(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(os, "environ", {"GROQ_API_KEY": "test-secret"})
    calls = fake_curl(monkeypatch)
    settings = provider.DeepSeekSettings.from_env(tmp_path)
    with_client = provider.DeepSeekClient(settings)
    try:
        result = with_client.request_json([{"role": "user", "content": 'JSON: «тест»\n"ok"'}],
                                          stage=stage, debug_dir=tmp_path / "debug")
    finally:
        with_client.close()
    assert result == {"tasks": []} and result.provenance["provider"] == "groq"
    assert with_client.counter.total == len(calls) == 1
    args, kwargs, options, headers = calls[0]
    assert args == ["curl", "--disable", "--config", "-"]
    assert "test-secret" not in repr(args)
    assert "Authorization: Bearer test-secret" in headers
    assert "Accept-Encoding: identity" in headers
    assert not any(header.lower().startswith("user-agent:") for header in headers)
    payload = json.loads(options["data-binary"])
    assert payload["model"] == provider.GROQ_DEFAULT_MODELS[stage]
    assert payload["max_completion_tokens"] == settings.max_tokens
    assert "max_tokens" not in payload and "thinking" not in payload
    assert (payload.get("reasoning_effort") == "none") == (stage == "ocr")
    assert settings.identity(stage)["free_providers"][0]["model"] == payload["model"]
    if stage == "ocr":
        assert settings.identity(stage)["free_providers"][0]["reasoning_effort"] == "none"
    assert float(options["connect-timeout"]) == 15 and float(options["max-time"]) == 135
    assert options["retry"] == "0" and "location" not in options
    assert not Path(options["output"]).exists()
    assert "test-secret" not in json.dumps(with_client.attempts)


@pytest.mark.parametrize("status,code,kind", [(429, 0, "http"), (503, 0, "http"),
                                             (0, 28, "timeout"), (0, 7, "network")])
def test_curl_failure_falls_back_and_cools_down_without_retry(tmp_path, monkeypatch, status, code, kind):
    monkeypatch.setattr(os, "environ", {"GROQ_API_KEY": "test-secret", "DEEPSEEK_API_KEY": "test-paid"})
    calls = fake_curl(monkeypatch, status, code)
    client = provider.DeepSeekClient(provider.DeepSeekSettings.from_env(tmp_path))
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}'
    paid_calls = []

    class PaidAdapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            paid_calls.append(request)
            return response
        def close(self):
            pass

    client.session.mount("https://api.deepseek.com/", PaidAdapter())
    try:
        for stage in ("ocr", "classification"):
            result = client.request_json([], stage=stage, debug_dir=tmp_path / "debug")
            assert result.provenance["provider"] == "deepseek-direct"
    finally:
        client.close()
    assert [a["provider"] for a in client.attempts] == ["groq", "deepseek-direct", "deepseek-direct"]
    assert client.attempts[0]["error_kind"] == kind
    assert len(calls) == 1 and len(paid_calls) == 2 and client.counter.total == 3


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("curl", 1), FileNotFoundError("private")])
def test_curl_process_errors_hide_details(monkeypatch, error):
    def run(*args, **kwargs):
        raise error
    monkeypatch.setattr(curl_transport.subprocess, "run", run)
    request = requests.Request("POST", "https://api.groq.com/openai/v1/chat/completions", json={}).prepare()
    with pytest.raises(requests.RequestException) as failure:
        curl_transport.CurlAdapter().send(request, timeout=(1, 2))
    assert "private" not in str(failure.value)


@pytest.mark.parametrize("value", ["", "custom-model"])
def test_groq_explicit_model_overrides_default(tmp_path, monkeypatch, value):
    monkeypatch.setattr(os, "environ", {"GROQ_API_KEY": "test-key", "GROQ_OCR_MODEL": value})
    selected, = provider.DeepSeekSettings.from_env(tmp_path).free_providers
    assert selected.ocr_model == value
    assert selected.classification_model == "openai/gpt-oss-120b"


def test_normal_pipeline_completes_both_stages_through_groq_curl(tmp_path, monkeypatch):
    import test_pipeline as fixtures
    from alexlarin_parser import pipeline

    monkeypatch.setattr(os, "environ", {"GROQ_API_KEY": "test-secret", "OCR_DPI": "72"})
    base, contexts, _ = fixtures.project.__wrapped__(tmp_path, monkeypatch)
    reference = fixtures.FakeClient()

    def response_factory(payload):
        messages = payload["messages"]
        stage = "ocr" if isinstance(messages[0]["content"], list) else "classification"
        assert payload["model"] == provider.GROQ_DEFAULT_MODELS[stage]
        data = reference.request_json(messages, stage=stage, debug_dir=contexts[0].debug_dir)
        return fixtures.completion(data).json()

    calls = fake_curl(monkeypatch, response_factory=response_factory)
    report = pipeline.run_batch(base, contexts[:1], local_only=True)
    state = report["variants"][0]
    assert state["error"] is None and state["result_current_attempt"] is True
    assert state["ocr_provenance"]["provider"] == state["classification_provenance"]["provider"] == "groq"
    assert len(calls) == report["ocr_calls"] + report["classification_calls"] == 3
    assert all(attempt["provider"] == "groq" for attempt in state["attempts"])


def test_groq_image_detail_is_adapted_without_mutating_fallback_input(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {"GROQ_API_KEY": "test-secret", "DEEPSEEK_API_KEY": "paid-key"})
    calls = fake_curl(monkeypatch, status=400)
    client = provider.DeepSeekClient(provider.DeepSeekSettings.from_env(tmp_path))
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "OCR JSON"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc", "detail": "original"}},
        {"type": "image_url", "image_url": {"url": "https://example.test/image", "detail": "low"}},
    ]}]
    original = json.loads(json.dumps(messages))
    paid_payloads = []

    class PaidAdapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            paid_payloads.append(json.loads(request.body))
            response = requests.Response()
            response.status_code = 200
            response._content = b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}'
            return response
        def close(self):
            pass

    client.session.mount("https://api.deepseek.com/", PaidAdapter())
    try:
        client.request_json(messages, stage="ocr", debug_dir=tmp_path / "debug")
    finally:
        client.close()
    sent = json.loads(calls[0][2]["data-binary"])["messages"]
    assert sent[0]["content"][1]["image_url"]["detail"] == "high"
    assert sent[0]["content"][2] == original[0]["content"][2]
    assert messages == original == paid_payloads[0]["messages"]
    identity = client.settings.identity("ocr")["free_providers"][0]
    assert identity["image_detail"] == "high"
    assert "image_detail" not in client.settings.identity("classification")["free_providers"][0]


def test_groq_oversize_tpm_preserves_diagnostics_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {"GROQ_API_KEY": "test-secret", "OPENROUTER_API_KEY": "free-key"})
    error = {"error": {"message": "TPM: Limit 8000, Requested 25790 test-secret",
                       "type": "tokens", "code": "rate_limit_exceeded", "private": "omit"}}
    calls = fake_curl(monkeypatch, status=413, response_factory=lambda payload: error)
    client = provider.DeepSeekClient(provider.DeepSeekSettings.from_env(tmp_path))
    # Exercise the same adapter path, including header parsing, with no network.
    original_run = curl_transport.subprocess.run
    def run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        path = Path(calls[-1][2]["dump-header"])
        with path.open('a') as stream:
            stream.write('x-ratelimit-limit-tokens: 8000\n'
                         'x-ratelimit-remaining-tokens: 8000\n'
                         'x-ratelimit-reset-tokens: 1ms\n'
                         'Set-Cookie: private\n')
        return result
    monkeypatch.setattr(curl_transport.subprocess, "run", run)
    class FreeAdapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            response = requests.Response()
            response.status_code = 200
            response._content = b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}'
            return response
        def close(self):
            pass
    client.session.mount("https://openrouter.ai/", FreeAdapter())
    try:
        result = client.request_json([], stage="classification", debug_dir=tmp_path / "debug")
    finally:
        client.close()
    assert result.provenance["provider"] == "openrouter"
    assert len(calls) == 1 and client.counter.total == 2
    attempt = client.attempts[0]
    assert attempt["api_error"]["code"] == "rate_limit_exceeded"
    assert set(attempt["api_error"]) == {"message", "type", "code"}
    assert attempt["rate_limit_headers"]["x-ratelimit-limit-tokens"] == "8000"
    assert attempt["rate_limit_headers"]["retry-after"] == "37"
    assert "set-cookie" not in attempt["rate_limit_headers"]
    assert "Requested 25790" in client.provider_diagnostics[2]["reason"]
    assert "test-secret" not in json.dumps(client.attempts + client.provider_diagnostics)
