"""Последовательная цепочка провайдеров OCR и классификации.

Клиент проверяет квоты перед каждой HTTP-попыткой, оболочку ответа
и учитывает вызовы. Проверка схем задач, кэши и решение о возобновлении остаются
в вызывающих этапах. Секрет не входит в identity и диагностические файлы.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import requests
from requests.adapters import HTTPAdapter

from .requests_counter import AIRequestCounter
from .io import atomic_json
from .rate_limits import ProviderLimits, RequestRateLimiter, QuotaExceeded
from email.utils import parsedate_to_datetime


def load_env(path):
    """Добавить KEY=value из файла в окружение, сохранив уже заданные ключи.

    Поддерживаются export, внешние кавычки и комментарии; shell-команды
    и подстановка переменных не исполняются. Отсутствующий файл пропускается,
    прочие ошибки чтения распространяются. Сам файл не изменяется.
    """
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip().isidentifier():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key.strip(), value)

_FREE_ENDPOINTS = (
    ("gemini", "GEMINI", "https://generativelanguage.googleapis.com/v1beta/openai"),
    ("nvidia-nim", "NVIDIA", "https://integrate.api.nvidia.com/v1"),
    ("groq", "GROQ", "https://api.groq.com/openai/v1"),
    ("openrouter", "OPENROUTER", "https://openrouter.ai/api/v1"),
)


def _load_rate_limits():
    limits = []
    prefixes = [(name, prefix) for name, prefix, _ in _FREE_ENDPOINTS]
    for name, prefix in [*prefixes, ("deepseek-direct", "DEEPSEEK")]:
        values = {}
        for kind in ("RPM", "RPD"):
            value = os.environ.get(f"{prefix}_{kind}", "").strip()
            if value:
                try:
                    values[kind.lower()] = int(value)
                except ValueError as error:
                    raise ValueError(f"{prefix}_{kind}: нужно положительное целое число") from error
        if values:
            limits.append(ProviderLimits(name, **values))
    return tuple(limits)


@dataclass(frozen=True)
class _FreeProvider:
    name: str
    base_url: str
    api_key: str = field(repr=False)
    ocr_model: str
    classification_model: str

    def model_for(self, stage):
        return self.ocr_model if stage == "ocr" else self.classification_model


def _load_free_providers():
    providers = []
    for name, prefix, base_url in _FREE_ENDPOINTS:
        api_key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        ocr_model = os.environ.get(f"{prefix}_OCR_MODEL", "").strip()
        classification_model = os.environ.get(
            f"{prefix}_CLASSIFICATION_MODEL", ""
        ).strip()
        if not api_key or not (ocr_model or classification_model):
            continue

        if name == "openrouter" and any(
            model and not model.endswith(":free")
            for model in (ocr_model, classification_model)
        ):
            raise ValueError("Модели OpenRouter должны иметь суффикс :free")

        providers.append(
            _FreeProvider(name, base_url, api_key, ocr_model, classification_model)
        )
    return tuple(providers)


def _retry_delay(headers):
    value = headers.get("Retry-After")
    if not value:
        return 60

    try:
        return max(1, int(value))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(1, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return 60


def _redact(value, api_key):
    if isinstance(value, str):
        return value.replace(api_key, "[REDACTED]")
    if isinstance(value, list):
        return [_redact(item, api_key) for item in value]
    if isinstance(value, dict):
        return {
            _redact(key, api_key): _redact(item, api_key)
            for key, item in value.items()
        }
    return value


@dataclass(frozen=True)
class DeepSeekSettings:
    """Неизменяемые настройки транспорта и рендеринга OCR.

    Тайм-ауты заданы в секундах, dpi — разрешение изображений страниц.
    Thinking OCR всегда отключён, классификация имеет отдельную настройку.
    Ключ скрыт из repr; публичная identity содержит только параметры модели,
    влияющие на совместимость кэшей. DPI учитывается этапом OCR отдельно.
    """
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    connect_timeout: float = 15
    read_timeout: float = 120
    classification_thinking: str = "disabled"
    max_tokens: int = 16384
    dpi: int = 300
    free_providers: tuple[_FreeProvider, ...] = ()
    rate_limits: tuple[ProviderLimits, ...] = ()
    rate_limit_path: Path | None = None


    def __post_init__(self):
        """Отклонить неверный thinking, неположительные лимиты и не-HTTPS URL."""
        if self.classification_thinking not in {"disabled", "enabled"}:
            raise ValueError("DEEPSEEK_CLASSIFICATION_THINKING: disabled либо enabled")
        if min(self.connect_timeout, self.read_timeout, self.max_tokens, self.dpi) <= 0:
            raise ValueError("Тайм-ауты, DPI и max_tokens должны быть положительными")
        if not self.base_url.startswith("https://"):
            raise ValueError("DEEPSEEK_BASE_URL должен использовать HTTPS")
        names = [limit.provider for limit in self.rate_limits]
        if len(names) != len(set(names)) or set(names) - {"deepseek-direct", *(p[0] for p in _FREE_ENDPOINTS)}:
            raise ValueError("Квоты должны задаваться один раз для каждого известного провайдера")

    @classmethod
    def from_env(cls, base_dir=None):
        """Собрать настройки транспорта, провайдеров и квот без обращения к API.

        Если base_dir задан, сначала прочитать его .env через load_env;
        существующие переменные окружения сохраняют приоритет. Неверные
        числа и значения настроек вызывают ValueError при создании объекта.
        """
        if base_dir is not None:
            load_env(Path(base_dir) / ".env")
        rate_limits = _load_rate_limits()
        return cls(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", cls.base_url).rstrip("/"),
            model=os.environ.get("DEEPSEEK_MODEL", cls.model),
            connect_timeout=float(os.environ.get("DEEPSEEK_CONNECT_TIMEOUT", "15")),
            read_timeout=float(os.environ.get("DEEPSEEK_READ_TIMEOUT",
                                               os.environ.get("DEEPSEEK_REQUEST_TIMEOUT", "120"))),
            classification_thinking=os.environ.get("DEEPSEEK_CLASSIFICATION_THINKING", "disabled"),
            max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "16384")),
            dpi=int(os.environ.get("OCR_DPI", "300")),
            free_providers=_load_free_providers(),
            rate_limits=rate_limits,
            rate_limit_path=(Path(base_dir) / "input" / "cache" / "provider_quotas.json"
                             if base_dir is not None and rate_limits else None),
        )

    def identity(self, stage):
        """Вернуть параметры провайдера для OCR или классификации без ключа.

        Поле thinking для OCR фиксировано, для классификации берётся
        из настроек. Тайм-ауты не влияют на содержимое ответа; DPI хранится
        отдельно в схеме рендеринга страницы, поэтому этих полей здесь нет.
        """
        identity = {
            "provider": "deepseek-direct", "base_url": self.base_url,
            "model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
            "thinking": "disabled" if stage == "ocr" else self.classification_thinking,
            "response_format": {"type": "json_object"},
        }
        free = [
            {"provider": provider.name, "base_url": provider.base_url, "model": model}
            for provider in self.free_providers
            if (model := provider.model_for(stage))
        ]
        if free:
            identity["free_providers"] = free
        return identity


class ProviderError(RuntimeError):
    """Ошибка сети, HTTP или оболочки ответа без автоматического повтора."""


class PaidCallsBlocked(ProviderError):
    """Блокировка платного DeepSeek; бесплатные провайдеры остаются доступны."""


class RateLimitBlocked(ProviderError):
    """Локальная квота исчерпана; HTTP-попытки и ожидания не было."""


class DeepSeekClient:
    """Общий транспорт OCR и классификации с последовательным fallback."""

    def __init__(self, settings, *, session=None, counter=None, rate_limiter=None):
        self.settings = settings
        self.session = session if session is not None else requests.Session()
        if session is None:
            self.session.mount("https://", HTTPAdapter(max_retries=0))
        self.counter = counter if counter is not None else AIRequestCounter()
        self.attempts = []
        self.blocked = None
        self._disabled = set()
        self._cooldown_until = {}
        self._request_lock = Lock()
        self._rate_limiter = rate_limiter if rate_limiter is not None else RequestRateLimiter(settings.rate_limit_path)

    def close(self):
        with self._request_lock:
            self.session.close()

    def request_json(self, messages, *, stage, debug_dir):
        # Один транспорт/журнал клиента: не допускаем гонок счётчика и сессии,
        # даже если внешний код вызовет клиент из нескольких потоков.
        with self._request_lock:
            return self._request_json(messages, stage=stage, debug_dir=debug_dir)

    def _request_json(self, messages, *, stage, debug_dir):
        if stage not in {"ocr", "classification"}:
            raise ValueError("Неизвестный этап AI")

        quota_errors = []
        for provider in self.settings.free_providers:
            model = provider.model_for(stage)
            if not model or provider.name in self._disabled:
                continue
            if time.monotonic() < self._cooldown_until.get(provider.name, 0):
                continue

            try:
                return self._request_once(
                    messages,
                    stage=stage,
                    debug_dir=debug_dir,
                    provider=provider.name,
                    base_url=provider.base_url,
                    model=model,
                    api_key=provider.api_key,
                )
            except RateLimitBlocked as error:
                quota_errors.append(str(error))
                continue
            except ProviderError:
                # Причина уже сохранена в журнале HTTP-попытки.
                continue

        if self.blocked:
            if quota_errors:
                raise RateLimitBlocked("; ".join([*quota_errors, self.blocked]))
            raise PaidCallsBlocked(self.blocked)
        if not self.settings.api_key:
            self.blocked = "Не задан DEEPSEEK_API_KEY; платные запросы не отправлялись"
            if quota_errors:
                raise RateLimitBlocked("; ".join([*quota_errors, self.blocked]))
            raise PaidCallsBlocked(self.blocked)

        return self._request_once(
            messages,
            stage=stage,
            debug_dir=debug_dir,
            provider="deepseek-direct",
            base_url=self.settings.base_url,
            model=self.settings.model,
            api_key=self.settings.api_key,
        )

    def _request_once(
        self, messages, *, stage, debug_dir, provider, base_url, model, api_key
    ):
        paid = provider == "deepseek-direct"
        token_field = "max_completion_tokens" if provider == "groq" else "max_tokens"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            token_field: self.settings.max_tokens,
            "stream": False,
        }
        if provider != "nvidia-nim":
            payload["response_format"] = {"type": "json_object"}
        if paid:
            thinking = (
                "disabled" if stage == "ocr" else self.settings.classification_thinking
            )
            payload["thinking"] = {"type": thinking}

        json.dumps(payload, ensure_ascii=False)
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        attempt = {
            "attempt": len(self.attempts) + 1,
            "stage": stage,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "status": "STARTED",
            "usage": None,
            "cost": None,
        }
        suffix = f"{attempt['attempt']:03d}_{stage}.json"
        attempt_path = debug_dir / f"attempt_{suffix}"
        limits = next((limit for limit in self.settings.rate_limits if limit.provider == provider), ProviderLimits(provider))
        try:
            with self._rate_limiter.reserve(limits, base_url=base_url, api_key=api_key):
                atomic_json(attempt_path, attempt)
        except QuotaExceeded as error:
            raise RateLimitBlocked(str(error)) from error
        self.attempts.append(attempt)
        setattr(self.counter, stage, getattr(self.counter, stage) + 1)
        started = time.monotonic()

        try:
            response = self.session.post(
                base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + api_key, "Accept": "application/json"},
                json=payload,
                timeout=(self.settings.connect_timeout, self.settings.read_timeout),
                allow_redirects=False,
            )
            status = response.status_code
            attempt["http_status"] = status

            if status in {401, 402, 403}:
                if paid:
                    self.blocked = f"DeepSeek HTTP {status}: проверьте ключ, доступ и баланс"
                else:
                    self._disabled.add(provider)
            elif not paid and (status in {408, 429} or 500 <= status < 600):
                self._cooldown_until[provider] = (
                    time.monotonic() + _retry_delay(response.headers)
                )

            try:
                body = response.json()
            except ValueError as error:
                if paid and self.blocked:
                    raise PaidCallsBlocked(self.blocked) from error
                raise ProviderError(f"{provider} HTTP {status}: ответ не JSON") from error

            atomic_json(debug_dir / f"response_{suffix}", _redact(body, api_key))
            if isinstance(body, dict):
                attempt["usage"] = numeric_usage(body.get("usage"))
            if paid and self.blocked:
                raise PaidCallsBlocked(self.blocked)
            if not 200 <= status < 300:
                raise ProviderError(f"{provider} HTTP {status}")

            choices = body.get("choices") if isinstance(body, dict) else None
            if not isinstance(choices, list) or len(choices) != 1:
                raise ProviderError(f"{provider}: нужен один элемент в choices")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ProviderError(f"{provider}: неверная структура choices")

            reason = choice.get("finish_reason")
            allowed = {
                "stop", "length", "content_filter", "tool_calls",
                "insufficient_system_resource",
            }
            reason = reason if isinstance(reason, str) and reason in allowed else "unknown"
            attempt["finish_reason"] = reason
            if reason != "stop":
                raise ProviderError(f"{provider}: неполный ответ, finish_reason={reason}")

            message = choice.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise ProviderError(f"{provider}: пустой content")
            try:
                data = json.loads(content)
            except ValueError as error:
                raise ProviderError(f"{provider}: content не является валидным JSON") from error
            if not isinstance(data, dict):
                raise ProviderError(f"{provider}: корень JSON должен быть объектом")

            attempt["status"] = "RECEIVED"
            return data
        except requests.RequestException as error:
            if not paid and isinstance(error, (requests.Timeout, requests.ConnectionError)):
                self._cooldown_until[provider] = time.monotonic() + 60
            attempt["status"] = "FAILED"
            attempt["error"] = type(error).__name__
            raise ProviderError(f"{provider}: {type(error).__name__}") from error
        except Exception as error:
            attempt["status"] = "FAILED"
            attempt["error"] = _redact(str(error), api_key)
            raise
        finally:
            attempt["seconds"] = round(time.monotonic() - started, 3)
            atomic_json(attempt_path, attempt)

def numeric_usage(value):
    """Вернуть известные неотрицательные целые счётчики токенов или None.

    Допускается частичный usage. Неизвестные поля и неверные значения,
    включая bool, отбрасываются; входной словарь не изменяется.
    """
    if not isinstance(value, dict):
        return None
    result = {key: value[key] for key in (
        "prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
    ) if type(value.get(key)) is int and value[key] >= 0}
    return result or None


def usage_summary(attempts):
    """Суммировать известные токены и посчитать попытки с неполным usage.

    При неизвестных данных суммы отражают только доступные поля, а
    unknown_usage_attempts показывает неполноту. Стоимость остаётся None:
    отсутствие usage или тарифа не превращается в нулевую стоимость.
    """
    known = [u for a in attempts if (u := numeric_usage(a.get("usage"))) is not None]
    return {
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in known),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in known),
        "total_tokens": sum(u.get("total_tokens", 0) for u in known),
        "unknown_usage_attempts": len(attempts) - sum(
            {"prompt_tokens", "completion_tokens", "total_tokens"}.issubset(u) for u in known
        ),
        "cost": None,
    }
