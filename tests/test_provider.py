"""Прямой DeepSeek: реальные HTTP-попытки, контракт и безопасные диагностики.

Все ответы подменены; эти тесты никогда не обращаются к сети и не читают
рабочий .env. Строка TEST_KEY заведомо фиктивна.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import requests

from ai_requests import AIRequestCounter
from provider import DeepSeekClient, DeepSeekSettings, PaidCallsBlocked, ProviderError, load_env, usage_summary
from alexlarin_parser import provider
from alexlarin_parser.rate_limits import ProviderLimits, RequestRateLimiter


TEST_KEY = "unit-test-not-a-real-api-key"
MESSAGES = [{"role": "user", "content": "Верни JSON-объект."}]


def body(*, content='{"tasks": []}', finish="stop", usage=True):
    result = {"choices": [{"finish_reason": finish, "message": {"content": content}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    return result


class FakeResponse:
    def __init__(self, status=200, data=None, *, invalid_json=False, headers=None):
        self.status_code = status
        self.data = body() if data is None else data
        self.invalid_json = invalid_json
        self.headers = headers or {}

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


def test_failed_preparation_is_not_a_paid_attempt(tmp_path, monkeypatch):
    import provider
    client, session, counter = client_for(FakeResponse())
    def fail_write(*args, **kwargs):
        raise OSError("нет места для диагностики")
    monkeypatch.setattr(provider, "atomic_json", fail_write)
    with pytest.raises(OSError):
        client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    assert not session.calls and not client.attempts and counter.total == 0


def test_partial_usage_remains_explicitly_unknown():
    result = usage_summary([{"usage": {"prompt_tokens": 12}}])
    assert result["prompt_tokens"] == 12
    assert result["unknown_usage_attempts"] == 1 and result["cost"] is None


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


@pytest.fixture
def chain_settings(tmp_path, monkeypatch):
    # Ни ключи процесса, ни рабочий .env не участвуют в проверке.
    monkeypatch.setattr(os, "environ", {})
    for prefix in ("OPENROUTER", "GROQ", "NVIDIA", "GEMINI"):
        monkeypatch.setenv(f"{prefix}_API_KEY", f"test-{prefix}-key")
        for stage in ("OCR", "CLASSIFICATION"):
            model = f"test-{prefix}-{stage}"
            monkeypatch.setenv(f"{prefix}_{stage}_MODEL", model + (":free" if prefix == "OPENROUTER" else ""))
    return DeepSeekSettings.from_env(tmp_path)


@pytest.mark.parametrize("stage", ["ocr", "classification"])
@pytest.mark.parametrize("successful_index", range(4))
def test_chain_order_first_success_payload_and_actual_attempts(tmp_path, chain_settings, stage, successful_index):
    from dataclasses import replace

    settings = replace(chain_settings, api_key=TEST_KEY, classification_thinking="enabled")
    session = FakeSession(*[FakeResponse(status=503) for _ in range(successful_index)], FakeResponse())
    client = DeepSeekClient(settings, session=session)
    assert client.request_json(MESSAGES, stage=stage, debug_dir=tmp_path) == {"tasks": []}
    order = ["gemini", "nvidia-nim", "openrouter", "deepseek-direct"]
    assert [attempt["provider"] for attempt in client.attempts] == order[:successful_index + 1]
    assert len(session.calls) == client.counter.total == getattr(client.counter, stage) == successful_index + 1
    assert [a["status"] for a in client.attempts] == ["FAILED"] * successful_index + ["RECEIVED"]
    endpoints = [p.base_url for p in settings.free_providers] + [settings.base_url]
    models = [p.model_for(stage) for p in settings.free_providers] + [settings.model]
    for name, endpoint, model, call in zip(order, endpoints, models, session.calls):
        assert call["url"] == endpoint + "/chat/completions"
        assert call["json"]["model"] == model
        assert call["allow_redirects"] is False
        assert call["timeout"] == (15, 120)
        if name == "deepseek-direct":
            assert call["json"]["thinking"] == {"type": "disabled" if stage == "ocr" else "enabled"}
        else:
            assert "thinking" not in call["json"]
        assert ("max_completion_tokens" in call["json"]) == (name == "groq")
        assert ("max_tokens" in call["json"]) == (name != "groq")
        assert ("response_format" in call["json"]) == (name != "nvidia-nim")


def test_stage_specific_free_models_do_not_require_deepseek_key(tmp_path, chain_settings, monkeypatch):
    monkeypatch.delenv("GEMINI_CLASSIFICATION_MODEL")
    monkeypatch.delenv("NVIDIA_OCR_MODEL")
    settings = DeepSeekSettings.from_env(tmp_path)
    session = FakeSession(FakeResponse(), FakeResponse())
    client = DeepSeekClient(settings, session=session)
    for stage in ("ocr", "classification"):
        assert client.request_json(MESSAGES, stage=stage, debug_dir=tmp_path) == {"tasks": []}
    assert not settings.api_key and client.blocked is None
    assert [a["provider"] for a in client.attempts] == ["gemini", "nvidia-nim"]
    assert client.counter.ocr == client.counter.classification == 1


@pytest.mark.parametrize("stage", ["ocr", "classification"])
@pytest.mark.parametrize("failed_index", range(3))
@pytest.mark.parametrize("failure", [429, 503, "timeout"])
def test_each_free_provider_falls_back_on_http_or_timeout(tmp_path, chain_settings, stage, failed_index, failure):
    settings = replace(chain_settings, api_key=TEST_KEY, free_providers=chain_settings.free_providers[failed_index:])
    outcome = requests.ReadTimeout("подставной тайм-аут") if failure == "timeout" else FakeResponse(status=failure)
    session = FakeSession(outcome, FakeResponse())
    client = DeepSeekClient(settings, session=session)
    response = client.request_json(MESSAGES, stage=stage, debug_dir=tmp_path)
    order = [p.name for p in settings.free_providers] + ["deepseek-direct"]
    assert [attempt["provider"] for attempt in client.attempts] == order[:2]
    assert [attempt["status"] for attempt in client.attempts] == ["FAILED", "RECEIVED"]
    assert response.provenance["provider"] == order[1]
    assert len(session.calls) == client.counter.total == getattr(client.counter, stage) == 2
    failed = next(event for event in client.provider_diagnostics if event["status"] == "FAILED")
    assert failed["kind"] == ("timeout" if failure == "timeout" else "http")


@pytest.mark.parametrize("missing_stage", ["ocr", "classification"])
def test_missing_nvidia_model_skips_only_that_stage(tmp_path, chain_settings, monkeypatch, missing_stage):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv(f"NVIDIA_{missing_stage.upper()}_MODEL", "")
    settings = DeepSeekSettings.from_env(tmp_path)
    session = FakeSession(FakeResponse())
    client = DeepSeekClient(settings, session=session)
    with pytest.raises(PaidCallsBlocked):
        client.request_json(MESSAGES, stage=missing_stage, debug_dir=tmp_path)
    assert not session.calls and not client.attempts and client.counter.total == 0
    available_stage = "classification" if missing_stage == "ocr" else "ocr"
    result = client.request_json(MESSAGES, stage=available_stage, debug_dir=tmp_path)
    assert result.provenance["provider"] == "nvidia-nim"
    assert [a["provider"] for a in client.attempts] == ["nvidia-nim"]
    assert len(session.calls) == client.counter.total == 1


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("retry_after,delay", [("37", 37), ("http-date", 90), (None, 60), ("invalid", 60)])
def test_cooldown_respects_retry_after_and_retries_after_expiry(
    tmp_path, chain_settings, monkeypatch, status, retry_after, delay,
):
    now = [100.0]
    monkeypatch.setattr(provider.time, "monotonic", lambda: now[0])

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 25, 12, tzinfo=timezone.utc)

    monkeypatch.setattr(provider, "datetime", FrozenDatetime)
    if retry_after == "http-date":
        retry_after = format_datetime(FrozenDatetime.now() + timedelta(seconds=90), usegmt=True)
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    session = FakeSession(FakeResponse(status=status, headers=headers), FakeResponse(), FakeResponse(), FakeResponse())
    client = DeepSeekClient(chain_settings, session=session, rate_limiter=RequestRateLimiter(clock=lambda: now[0]))
    client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    now[0] += delay - 0.1
    client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path)
    now[0] += 0.2
    client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    assert [a["provider"] for a in client.attempts] == ["gemini", "nvidia-nim", "nvidia-nim", "gemini"]
    assert len(session.calls) == client.counter.total == 4


def test_timeout_falls_back_and_cools_down_without_extra_attempts(tmp_path, chain_settings, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(provider.time, "monotonic", lambda: now[0])
    session = FakeSession(requests.Timeout("test timeout"), FakeResponse(), FakeResponse(), FakeResponse())
    client = DeepSeekClient(chain_settings, session=session, rate_limiter=RequestRateLimiter(clock=lambda: now[0]))
    client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    now[0] += 59
    client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    now[0] += 2
    client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    assert [a["provider"] for a in client.attempts] == ["gemini", "nvidia-nim", "nvidia-nim", "gemini"]
    assert len(session.calls) == client.counter.ocr == 4
    assert client.attempts[0]["error"] == "Timeout"


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("failed_index", range(3))
def test_auth_failure_disables_only_current_free_provider(tmp_path, chain_settings, status, failed_index):
    from dataclasses import replace

    settings = replace(chain_settings, api_key=TEST_KEY, free_providers=chain_settings.free_providers[failed_index:])
    session = FakeSession(FakeResponse(status=status, invalid_json=True), FakeResponse(), FakeResponse())
    client = DeepSeekClient(settings, session=session)
    for stage in ("ocr", "classification"):
        client.request_json(MESSAGES, stage=stage, debug_dir=tmp_path)
    order = [p.name for p in settings.free_providers] + ["deepseek-direct"]
    assert [a["provider"] for a in client.attempts] == [order[0], order[1], order[1]]
    assert client._disabled == {order[0]} and client.blocked is None
    assert len(session.calls) == client.counter.total == 3


def test_failed_chain_without_paid_key_counts_only_actual_requests(tmp_path, chain_settings):
    session = FakeSession(*[FakeResponse(status=503) for _ in range(3)])
    client = DeepSeekClient(chain_settings, session=session)
    with pytest.raises(PaidCallsBlocked):
        client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    with pytest.raises(PaidCallsBlocked):
        client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path)
    assert len(session.calls) == client.counter.total == len(client.attempts) == 3
    assert client.counter.classification == 0


@pytest.mark.parametrize("model", ["paid-model", "", "provider/model:free"])
def test_inactive_environment_models_and_quotas_do_not_affect_route(tmp_path, chain_settings, monkeypatch, model):
    for prefix in ("GROQ",):
        monkeypatch.setenv(f"{prefix}_OCR_MODEL", model)
        monkeypatch.setenv(f"{prefix}_RPM", "invalid")
        monkeypatch.setenv(f"{prefix}_RPD", "0")
    settings = DeepSeekSettings.from_env(tmp_path)
    assert [p.name for p in settings.free_providers] == ["gemini", "nvidia-nim", "openrouter"]
    assert not settings.rate_limits
    assert all(settings.identity(stage) == chain_settings.identity(stage) for stage in ("ocr", "classification"))


@pytest.mark.parametrize("model", ["paid-model", "", "provider/model:free"])
def test_openrouter_accepts_only_explicit_free_models(tmp_path, chain_settings, monkeypatch, model):
    monkeypatch.setenv("OPENROUTER_OCR_MODEL", model)
    if model == "paid-model":
        with pytest.raises(ValueError, match=":free"):
            DeepSeekSettings.from_env(tmp_path)
    else:
        settings = DeepSeekSettings.from_env(tmp_path)
        assert settings.free_providers[-1].name == "openrouter"
        assert settings.free_providers[-1].ocr_model == model


@pytest.mark.parametrize("limit", ["RPM", "RPD"])
def test_configured_provider_quota_blocks_http_before_attempt_counter(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(os, "environ", {"DEEPSEEK_API_KEY": TEST_KEY, f"DEEPSEEK_{limit}": "1"})
    session = FakeSession(FakeResponse(), FakeResponse())
    client = DeepSeekClient(DeepSeekSettings.from_env(tmp_path), session=session)
    client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path / "debug")
    with pytest.raises(ProviderError, match=limit):
        client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path / "debug")
    assert len(session.calls) == client.counter.total == len(client.attempts) == 1
    assert client.counter.classification == 0


@pytest.mark.parametrize("kind,window", [("rpm", 60), ("rpd", 86400)])
def test_quota_reopens_only_at_rolling_window_boundary(tmp_path, kind, window):
    now = [100000.0]
    settings = DeepSeekSettings(api_key=TEST_KEY, rate_limits=(ProviderLimits("deepseek-direct", **{kind: 1}),))
    session = FakeSession(FakeResponse(), FakeResponse())
    client = DeepSeekClient(settings, session=session, rate_limiter=RequestRateLimiter(clock=lambda: now[0]))
    client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    now[0] += window - 0.01
    with pytest.raises(provider.RateLimitBlocked, match=kind.upper()):
        client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path)
    now[0] += 0.01
    client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path)
    assert len(session.calls) == client.counter.total == 2


def test_daily_budget_still_blocks_after_minute_budget_recovers(tmp_path):
    now = [100000.0]
    settings = DeepSeekSettings(api_key=TEST_KEY, rate_limits=(ProviderLimits("deepseek-direct", rpm=1, rpd=2),))
    session = FakeSession(FakeResponse(), FakeResponse(), FakeResponse())
    client = DeepSeekClient(settings, session=session, rate_limiter=RequestRateLimiter(clock=lambda: now[0]))
    client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    now[0] += 60
    client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path)
    now[0] += 60
    with pytest.raises(provider.RateLimitBlocked, match="RPD"):
        client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    now[0] = 186400.0
    client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    assert len(session.calls) == client.counter.total == 3


def test_every_provider_including_paid_tail_has_independent_shared_stage_quota(tmp_path, chain_settings):
    names = [p.name for p in chain_settings.free_providers] + ["deepseek-direct"]
    settings = replace(chain_settings, api_key=TEST_KEY,
                       rate_limits=tuple(ProviderLimits(name, rpm=1, rpd=1) for name in names))
    session = FakeSession(*[FakeResponse() for _ in names])
    client = DeepSeekClient(settings, session=session, rate_limiter=RequestRateLimiter(clock=lambda: 100000.0))
    for stage in ("ocr", "classification", "ocr", "classification"):
        client.request_json(MESSAGES, stage=stage, debug_dir=tmp_path)
    with pytest.raises(provider.RateLimitBlocked):
        client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path)
    assert [a["provider"] for a in client.attempts] == names
    assert len(session.calls) == client.counter.total == 4
    assert (client.counter.ocr, client.counter.classification) == (2, 2)
    assert len(list(tmp_path.glob("attempt_*.json"))) == 4


def test_error_fallback_checks_target_provider_budget_too(tmp_path, chain_settings, monkeypatch):
    now = [100000.0]
    monkeypatch.setattr(provider.time, "monotonic", lambda: now[0])
    settings = replace(chain_settings, api_key=TEST_KEY, rate_limits=(ProviderLimits("nvidia-nim", rpd=1),))
    session = FakeSession(FakeResponse(status=503), FakeResponse(), FakeResponse(status=503), FakeResponse())
    client = DeepSeekClient(settings, session=session, rate_limiter=RequestRateLimiter(clock=lambda: now[0]))
    client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    now[0] += 61
    client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path)
    assert [a["provider"] for a in client.attempts] == ["gemini", "nvidia-nim", "gemini", "openrouter"]
    assert len(session.calls) == client.counter.total == 4


@pytest.mark.parametrize("outcome", [requests.Timeout("test timeout"), FakeResponse(status=503)])
def test_failed_http_attempt_consumes_quota(tmp_path, outcome):
    settings = DeepSeekSettings(api_key=TEST_KEY, rate_limits=(ProviderLimits("deepseek-direct", rpd=1),))
    session = FakeSession(outcome)
    client = DeepSeekClient(settings, session=session)
    with pytest.raises(ProviderError):
        client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    with pytest.raises(provider.RateLimitBlocked, match="RPD"):
        client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path)
    assert len(session.calls) == client.counter.total == 1


def test_failed_preparation_does_not_consume_quota(tmp_path, monkeypatch):
    settings = DeepSeekSettings(api_key=TEST_KEY, rate_limits=(ProviderLimits("deepseek-direct", rpd=1),))
    client = DeepSeekClient(settings, session=FakeSession(FakeResponse()))
    with monkeypatch.context() as patch:
        patch.setattr(provider, "atomic_json", lambda *a, **k: (_ for _ in ()).throw(OSError("нет места")))
        with pytest.raises(OSError, match="нет места"):
            client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path)
    assert client.counter.total == 0
    client.request_json(MESSAGES, stage="classification", debug_dir=tmp_path)
    assert client.counter.total == 1


def test_quota_history_survives_new_client_and_model_change_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {"DEEPSEEK_API_KEY": TEST_KEY, "DEEPSEEK_RPD": "1"})
    settings = DeepSeekSettings.from_env(tmp_path)
    first = DeepSeekClient(settings, session=FakeSession(FakeResponse()))
    first.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path / "first")
    first.close()
    second_session = FakeSession()
    second = DeepSeekClient(replace(settings, model="another-model"), session=second_session)
    with pytest.raises(provider.RateLimitBlocked, match="RPD"):
        second.request_json(MESSAGES, stage="classification", debug_dir=tmp_path / "second")
    assert second.counter.total == 0 and not second_session.calls
    assert TEST_KEY not in settings.rate_limit_path.read_text()


def test_same_account_budget_is_atomic_between_clients(tmp_path):
    settings = DeepSeekSettings(api_key=TEST_KEY, rate_limits=(ProviderLimits("deepseek-direct", rpd=1),),
                                rate_limit_path=tmp_path / "quotas.json")
    clients = [DeepSeekClient(settings, session=FakeSession(FakeResponse())) for _ in range(2)]
    start = Barrier(2)

    def request(index):
        start.wait(timeout=3)
        try:
            return clients[index].request_json(MESSAGES, stage="ocr", debug_dir=tmp_path / str(index))
        except provider.RateLimitBlocked:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(request, range(2)))
    assert results.count({"tasks": []}) == 1 and results.count(None) == 1
    assert sum(client.counter.total for client in clients) == 1


def test_parallel_calls_to_one_client_keep_unique_attempts_and_input_binding(tmp_path):
    class EchoSession(FakeSession):
        def post(self, url, **kwargs):
            self.calls.append({"url": url, **kwargs})
            return FakeResponse(data=body(content=json.dumps({"task": kwargs["json"]["messages"][0]["content"]})))

    settings = DeepSeekSettings(api_key=TEST_KEY, rate_limits=(ProviderLimits("deepseek-direct", rpd=2),))
    session = EchoSession()
    client = DeepSeekClient(settings, session=session)

    def request(index):
        try:
            return client.request_json([{"role": "user", "content": str(index)}], stage="ocr", debug_dir=tmp_path)
        except provider.RateLimitBlocked:
            return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(request, range(4)))
    assert sum(result is not None for result in results) == 2
    assert all(result is None or result == {"task": str(index)} for index, result in enumerate(results))
    assert [a["attempt"] for a in client.attempts] == [1, 2]
    assert len(session.calls) == client.counter.total == 2


@pytest.mark.parametrize("prefix", ["GEMINI", "NVIDIA", "OPENROUTER", "DEEPSEEK"])
@pytest.mark.parametrize("kind", ["RPM", "RPD"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_invalid_limit_configuration_is_rejected(tmp_path, monkeypatch, prefix, kind, value):
    monkeypatch.setattr(os, "environ", {f"{prefix}_{kind}": value})
    with pytest.raises(ValueError, match="положительное целое число"):
        DeepSeekSettings.from_env(tmp_path)


def test_limits_for_all_providers_are_loaded_without_changing_cache_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    initial = DeepSeekSettings.from_env(tmp_path)
    names = ("gemini", "nvidia-nim", "openrouter", "deepseek-direct")
    for index, prefix in enumerate(("GEMINI", "NVIDIA", "OPENROUTER", "DEEPSEEK"), 1):
        monkeypatch.setenv(f"{prefix}_RPM", str(index))
        monkeypatch.setenv(f"{prefix}_RPD", str(index * 10))
    settings = DeepSeekSettings.from_env(tmp_path)
    assert settings.rate_limits == tuple(ProviderLimits(name, rpm=i, rpd=i * 10) for i, name in enumerate(names, 1))
    assert all(settings.identity(stage) == initial.identity(stage) for stage in ("ocr", "classification"))
    assert not settings.rate_limit_path.exists()


def test_corrupt_quota_ledger_does_not_allow_http(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {"DEEPSEEK_API_KEY": TEST_KEY, "DEEPSEEK_RPD": "1"})
    settings = DeepSeekSettings.from_env(tmp_path)
    settings.rate_limit_path.parent.mkdir(parents=True)
    settings.rate_limit_path.write_text("{truncated")
    client = DeepSeekClient(settings, session=FakeSession())
    with pytest.raises(ValueError, match="Журнал квот повреждён"):
        client.request_json(MESSAGES, stage="ocr", debug_dir=tmp_path / "debug")
    assert client.counter.total == 0 and not client.attempts
