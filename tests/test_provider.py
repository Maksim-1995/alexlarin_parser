"""Прямой DeepSeek: реальные HTTP-попытки, контракт и безопасные диагностики.

Все ответы подменены; эти тесты никогда не обращаются к сети и не читают
рабочий .env. Строка TEST_KEY заведомо фиктивна.
"""

import json
import os

import pytest
import requests

from ai_requests import AIRequestCounter
from provider import DeepSeekClient, DeepSeekSettings, PaidCallsBlocked, ProviderError, load_env, usage_summary


TEST_KEY = "unit-test-not-a-real-api-key"
MESSAGES = [{"role": "user", "content": "Верни JSON-объект."}]


def body(*, content='{"tasks": []}', finish="stop", usage=True):
    result = {"choices": [{"finish_reason": finish, "message": {"content": content}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    return result


class FakeResponse:
    def __init__(self, status=200, data=None, *, invalid_json=False):
        self.status_code = status
        self.data = body() if data is None else data
        self.invalid_json = invalid_json

    def json(self):
        if self.invalid_json:
            raise ValueError("not JSON")
        return self.data


class FakeSession:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        pass


def client_for(*outcomes, **settings):
    counter = AIRequestCounter()
    session = FakeSession(*outcomes)
    client = DeepSeekClient(DeepSeekSettings(api_key=TEST_KEY, **settings), session=session, counter=counter)
    return client, session, counter


def test_one_client_uses_explicit_thinking_json_mode_timeouts_and_direct_endpoint(tmp_path):
    client, session, counter = client_for(FakeResponse(), FakeResponse(), classification_thinking="enabled",
                                        connect_timeout=7, read_timeout=23)
    assert client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path) == {"tasks": []}
    assert client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path) == {"tasks": []}
    assert counter.ocr == counter.classification == 1
    assert counter.total == 2
    for call, thinking in zip(session.calls, ("disabled", "enabled")):
        assert call["url"] == "https://api.deepseek.com/chat/completions"
        assert call["allow_redirects"] is False
        assert call["timeout"] == (7, 23)
        assert call["headers"]["Authorization"] == "Bearer " + TEST_KEY
        payload = call["json"]
        assert set(payload) == {"model", "messages", "thinking", "temperature", "response_format", "max_tokens", "stream"}
        assert payload["model"] == "deepseek-flash"
        assert payload["thinking"] == {"type": thinking}
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["stream"] is False
    assert [attempt["status"] for attempt in client.attempts] == ["RECEIVED", "RECEIVED"]
    assert usage_summary(client.attempts) == {"prompt_tokens": 24, "completion_tokens": 6, "total_tokens": 30,
                                             "unknown_usage_attempts": 0, "cost": None}


def test_default_requests_session_has_no_adapter_retries():
    client = DeepSeekClient(DeepSeekSettings(api_key=TEST_KEY))
    try:
        assert client.session.get_adapter("https://").max_retries.total == 0
    finally:
        client.close()


@pytest.mark.parametrize("outcome", [
    FakeResponse(status=500, data={"error": {"message": "server failure"}}),
    FakeResponse(status=302, data={"error": {"message": "redirect"}}),
    FakeResponse(invalid_json=True),
    FakeResponse(data=body(content="")),
    FakeResponse(data=body(content="{truncated")),
    FakeResponse(data=body(content="[]")),
    FakeResponse(data=body(finish="length")),
    FakeResponse(data={"choices": []}),
    FakeResponse(data={"choices": [None]}),
    requests.Timeout("read timed out"),
])
def test_each_unsuccessful_actual_attempt_is_counted_once_without_retry(tmp_path, outcome):
    client, session, counter = client_for(outcome)
    with pytest.raises(ProviderError):
        client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    assert len(session.calls) == counter.ocr == 1
    assert counter.classification == 0
    assert len(client.attempts) == 1
    attempt = client.attempts[0]
    assert attempt["status"] == "FAILED"
    assert attempt["seconds"] >= 0
    assert attempt["cost"] is None
    saved = json.loads((tmp_path / "attempt_001_ocr.json").read_text())
    assert saved == attempt


@pytest.mark.parametrize("status", [401, 402, 403])
@pytest.mark.parametrize("invalid_json", [False, True])
def test_common_authorization_or_balance_failure_blocks_later_paid_calls_even_for_html(tmp_path, status, invalid_json):
    client, session, counter = client_for(FakeResponse(status=status, data={"error": "blocked"}, invalid_json=invalid_json))
    with pytest.raises(PaidCallsBlocked):
        client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    with pytest.raises(PaidCallsBlocked):
        client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path)
    assert len(session.calls) == counter.total == 1


def test_missing_key_prevents_any_http_attempt_and_keeps_counter_zero(tmp_path):
    session = FakeSession()
    counter = AIRequestCounter()
    client = DeepSeekClient(DeepSeekSettings(api_key=""), session=session, counter=counter)
    for stage in ("ocr", "classification"):
        with pytest.raises(PaidCallsBlocked, match="DEEPSEEK_API_KEY"):
            client.request_json(MESSAGES, stage=stage, debug_dir=tmp_path / "debug")
    assert not session.calls
    assert counter.total == 0
    assert not client.attempts
    assert not (tmp_path / "debug").exists()


def test_unserializable_request_is_not_counted_as_http_attempt(tmp_path):
    client, session, counter = client_for()
    with pytest.raises(TypeError):
        client.request_json([{"content": object()}], stage="ocr", debug_dir=tmp_path)
    assert not session.calls
    assert counter.total == 0
    assert not client.attempts


def test_secret_is_hidden_from_settings_and_raw_error_diagnostics(tmp_path, capsys):
    client, session, counter = client_for(FakeResponse(status=500, data={
        "error": {"message": "echo " + TEST_KEY}, "usage": {"debug": TEST_KEY},
    }))
    assert TEST_KEY not in repr(client.settings)
    assert TEST_KEY not in json.dumps(client.settings.identity("ocr"))
    with pytest.raises(ProviderError) as failure:
        client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    assert TEST_KEY not in str(failure.value)
    assert TEST_KEY not in capsys.readouterr().out
    for path in tmp_path.glob("*.json"):
        assert TEST_KEY not in path.read_text()


def test_timeout_error_message_does_not_copy_request_exception_or_secret(tmp_path):
    client, _, _ = client_for(requests.Timeout("request details " + TEST_KEY))
    with pytest.raises(ProviderError) as failure:
        client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    assert TEST_KEY not in str(failure.value)
    assert TEST_KEY not in (tmp_path / "attempt_001_ocr.json").read_text()


def test_unknown_usage_and_unknown_price_are_not_reported_as_zero_cost(tmp_path):
    client, _, _ = client_for(FakeResponse(data=body(usage=False)), requests.Timeout("timeout"))
    client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    with pytest.raises(ProviderError):
        client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path)
    summary = usage_summary(client.attempts)
    assert summary["unknown_usage_attempts"] == 2
    assert summary["cost"] is None
    assert all(attempt["usage"] is None and attempt["cost"] is None for attempt in client.attempts)


def test_dotenv_is_parsed_as_text_without_execution_and_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    marker = tmp_path / "must_not_exist"
    env = tmp_path / ".env"
    env.write_text("\n".join([
        "# test configuration", "export TEST_DOTENV_VALUE='literal value'",
        "TEST_DOTENV_EXISTING=from-file", f"TEST_DOTENV_SHELL=$(touch {marker})",
        "TEST_DOTENV_COMMENT=ok # inline comment", "not a valid key=ignored", "bad line",
    ]))
    for key in ("TEST_DOTENV_VALUE", "TEST_DOTENV_SHELL", "TEST_DOTENV_COMMENT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TEST_DOTENV_EXISTING", "from-environment")
    load_env(env)
    assert os.environ["TEST_DOTENV_VALUE"] == "literal value"
    assert os.environ["TEST_DOTENV_EXISTING"] == "from-environment"
    assert os.environ["TEST_DOTENV_SHELL"] == f"$(touch {marker})"
    assert os.environ["TEST_DOTENV_COMMENT"] == "ok"
    assert not marker.exists()


def test_settings_use_only_explicit_environment_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "DEEPSEEK_CONNECT_TIMEOUT",
                 "DEEPSEEK_READ_TIMEOUT", "DEEPSEEK_REQUEST_TIMEOUT", "DEEPSEEK_CLASSIFICATION_THINKING",
                 "DEEPSEEK_MAX_TOKENS", "OCR_DPI"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(f"DEEPSEEK_API_KEY={TEST_KEY}\nDEEPSEEK_READ_TIMEOUT=55\n")
    monkeypatch.setenv("DEEPSEEK_READ_TIMEOUT", "31")
    settings = DeepSeekSettings.from_env(tmp_path)
    assert settings.api_key == TEST_KEY
    assert settings.read_timeout == 31
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model == "deepseek-flash"
    assert settings.identity("ocr")["thinking"] == settings.identity("classification")["thinking"] == "disabled"
