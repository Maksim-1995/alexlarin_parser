"""Общий прямой DeepSeek-транспорт для OCR и классификации.

Клиент отвечает за единственную HTTP-попытку, проверку оболочки ответа
и учёт вызовов. Проверка схем задач, кэши и решение о возобновлении остаются
в вызывающих этапах. Секрет не входит в identity и диагностические файлы.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

from .requests_counter import AIRequestCounter
from .io import atomic_json


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

    def __post_init__(self):
        """Отклонить неверный thinking, неположительные лимиты и не-HTTPS URL."""
        if self.classification_thinking not in {"disabled", "enabled"}:
            raise ValueError("DEEPSEEK_CLASSIFICATION_THINKING: disabled либо enabled")
        if min(self.connect_timeout, self.read_timeout, self.max_tokens, self.dpi) <= 0:
            raise ValueError("Тайм-ауты, DPI и max_tokens должны быть положительными")
        if not self.base_url.startswith("https://"):
            raise ValueError("DEEPSEEK_BASE_URL должен использовать HTTPS")

    @classmethod
    def from_env(cls, base_dir=None):
        """Собрать настройки DEEPSEEK_* и OCR_DPI без обращения к API.

        Если base_dir задан, сначала прочитать его .env через load_env;
        существующие переменные окружения сохраняют приоритет. Неверные
        числа и значения настроек вызывают ValueError при создании объекта.
        """
        if base_dir is not None:
            load_env(Path(base_dir) / ".env")
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
        )

    def identity(self, stage):
        """Вернуть параметры провайдера для OCR или классификации без ключа.

        Поле thinking для OCR фиксировано, для классификации берётся
        из настроек. Тайм-ауты не влияют на содержимое ответа; DPI хранится
        отдельно в схеме рендеринга страницы, поэтому этих полей здесь нет.
        """
        return {
            "provider": "deepseek-direct", "base_url": self.base_url,
            "model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
            "thinking": "disabled" if stage == "ocr" else self.classification_thinking,
            "response_format": {"type": "json_object"},
        }


class ProviderError(RuntimeError):
    """Ошибка сети, HTTP или оболочки ответа без автоматического повтора."""


class PaidCallsBlocked(ProviderError):
    """Общая ошибка ключа, доступа или баланса запрещает новые вызовы клиента."""


class DeepSeekClient:
    """Одна HTTP-сессия и общий журнал попыток для обоих AI-этапов.

    counter хранит число попыток по этапам, attempts — их длительность,
    статус и доступный usage. Это журнал текущего клиента, а не история
    кэшей. Ошибка авторизации/баланса сохраняет blocked: следующие запросы
    отклоняются до HTTP. Пайплайн закрывает сессию через close.
    """

    def __init__(self, settings, *, session=None, counter=None):
        """Сохранить настройки и счётчик; создать сессию без HTTP-повторов.

        Переданный session позволяет подменять транспорт в локальных тестах;
        его адаптеры не перенастраиваются. Без counter создаётся новый счётчик.
        """
        self.settings = settings
        self.session = session or requests.Session()
        if session is None:
            self.session.mount("https://", HTTPAdapter(max_retries=0))
        self.counter = counter or AIRequestCounter()
        self.attempts = []
        self.blocked = None

    def close(self):
        """Закрыть HTTP-сессию, включая явно переданную при создании клиента."""
        self.session.close()

    def request_json(self, messages, *, stage, debug_dir):
        """Отправить один chat/completions запрос и вернуть JSON-объект content.

        messages — готовые сообщения с текстом/изображениями; stage задаёт
        ocr или classification. Сериализация запроса и начальная запись
        диагностики выполняются до увеличения счётчика. Затем учитывается
        каждая HTTP-попытка, включая ошибку и тайм-аут, без скрытых повторов.

        debug_dir получает журнал попытки и ответ API без секретного ключа.
        HTTP-успех требует непустого JSON content и finish_reason=stop;
        статус RECEIVED ещё не подтверждает схему или точность задач.
        Их проверяет вызывающий этап перед записью своего кэша.

        PaidCallsBlocked означает отсутствующий ключ либо HTTP 401/402/403
        и блокирует следующие вызовы. ProviderError означает ошибку сети,
        HTTP или ответа. Неверный stage/несериализуемый запрос и ошибки
        записи диагностики также распространяются, не маскируясь успехом.
        """
        if stage not in {"ocr", "classification"}:
            raise ValueError("Неизвестный этап AI")
        if self.blocked:
            raise PaidCallsBlocked(self.blocked)
        if not self.settings.api_key:
            self.blocked = "Не задан DEEPSEEK_API_KEY; платные запросы не отправлялись"
            raise PaidCallsBlocked(self.blocked)
        identity = self.settings.identity(stage)
        payload = {
            "model": self.settings.model, "messages": messages,
            "thinking": {"type": identity["thinking"]}, "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": self.settings.max_tokens, "stream": False,
        }
        # Сериализация до счётчика: ошибка подготовки не считается HTTP-попыткой.
        json.dumps(payload, ensure_ascii=False)
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        attempt = {
            "attempt": len(self.attempts) + 1, "stage": stage,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "provider": identity["provider"], "model": self.settings.model,
            "status": "STARTED", "usage": None, "cost": None,
        }
        atomic_json(debug_dir / f"attempt_{attempt['attempt']:03d}_{stage}.json", attempt)
        self.attempts.append(attempt)
        setattr(self.counter, stage, getattr(self.counter, stage) + 1)
        started = time.monotonic()
        try:
            response = self.session.post(
                self.settings.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + self.settings.api_key,
                         "Accept": "application/json"},
                json=payload, timeout=(self.settings.connect_timeout, self.settings.read_timeout),
                allow_redirects=False,
            )
            attempt["http_status"] = response.status_code
            if response.status_code in {401, 402, 403}:
                self.blocked = f"DeepSeek HTTP {response.status_code}: проверьте ключ, доступ и баланс; пакет остановлен"
            try:
                body = response.json()
            except ValueError as error:
                if self.blocked:
                    raise PaidCallsBlocked(self.blocked) from error
                raise ProviderError(f"DeepSeek HTTP {response.status_code}: ответ не JSON") from error
            # Сохраняем только ответ, никогда запрос с Authorization/base64.
            raw = json.dumps(body, ensure_ascii=False).replace(self.settings.api_key, "[REDACTED]")
            atomic_json(debug_dir / f"response_{attempt['attempt']:03d}_{stage}.json", json.loads(raw))
            if isinstance(body, dict):
                attempt["usage"] = numeric_usage(body.get("usage"))
            if self.blocked:
                raise PaidCallsBlocked(self.blocked)
            if not 200 <= response.status_code < 300:
                raise ProviderError(f"DeepSeek HTTP {response.status_code}; автоматического повтора нет")
            if not isinstance(body, dict) or not isinstance(body.get("choices"), list) or len(body["choices"]) != 1:
                raise ProviderError("DeepSeek: нужен один choices в ответе")
            choice = body["choices"][0]
            if not isinstance(choice, dict):
                raise ProviderError("DeepSeek: неверная структура choices")
            reason = choice.get("finish_reason")
            reason = reason if isinstance(reason, str) and reason in {
                "stop", "length", "content_filter", "tool_calls", "insufficient_system_resource",
            } else "unknown"
            attempt["finish_reason"] = reason
            if reason != "stop":
                raise ProviderError(f"DeepSeek: неполный ответ, finish_reason={reason}")
            message = choice.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("DeepSeek: пустой content")
            try:
                data = json.loads(content)
            except ValueError as error:
                raise ProviderError("DeepSeek: content не является валидным JSON") from error
            if not isinstance(data, dict):
                raise ProviderError("DeepSeek: корень JSON должен быть объектом")
            attempt["status"] = "RECEIVED"
            return data
        except requests.RequestException as error:
            attempt["status"] = "FAILED"
            attempt["error"] = type(error).__name__
            raise ProviderError(f"DeepSeek: {type(error).__name__}; автоматического повтора нет") from error
        except Exception as error:
            attempt["status"] = "FAILED"
            attempt["error"] = str(error).replace(self.settings.api_key, "[REDACTED]")
            raise
        finally:
            attempt["seconds"] = round(time.monotonic() - started, 3)
            atomic_json(debug_dir / f"attempt_{attempt['attempt']:03d}_{stage}.json", attempt)


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
