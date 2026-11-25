"""Общие квоты провайдера на скользящие окна минуты и суток."""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
from pathlib import Path
from threading import Lock
import time

from .io import atomic_json, fingerprint


@dataclass(frozen=True)
class ProviderLimits:
    provider: str
    rpm: int | None = None
    rpd: int | None = None

    def __post_init__(self):
        for name, value in (("RPM", self.rpm), ("RPD", self.rpd)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{self.provider} {name}: нужно положительное целое число")


class QuotaExceeded(RuntimeError):
    def __init__(self, provider, exhausted, retry_after):
        self.retry_after = retry_after
        super().__init__(f"{provider}: локальный лимит {'/'.join(exhausted)}; "
                         f"повтор возможен через {math.ceil(retry_after)} с")


class RequestRateLimiter:
    """Резервировать квоту до HTTP, не ждать и не повторять запрос.

    С path журнал переживает смену клиента/процесса. Отдельный flock-файл
    защищает read/check/write, атомарная замена JSON не меняет inode блокировки.
    Без path история хранится в экземпляре. В журнале нет открытых API-ключей.
    """

    def __init__(self, path=None, *, clock=None):
        self.path = Path(path) if path is not None else None
        self.clock = clock if clock is not None else time.time
        self._lock = Lock()
        self._state = {"version": 1, "providers": {}}

    def _read(self):
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": 1, "providers": {}}
        except (ValueError, UnicodeError) as error:
            raise ValueError("Журнал квот повреждён; HTTP не отправлен") from error
        if not isinstance(state, dict) or state.get("version") != 1 or not isinstance(state.get("providers"), dict):
            raise ValueError("Неверный формат журнала квот; HTTP не отправлен")
        for record in state["providers"].values():
            if (not isinstance(record, dict) or not isinstance(record.get("requests"), list)
                    or any(type(t) not in {int, float} or not math.isfinite(t) or t < 0
                           for t in record["requests"])):
                raise ValueError("Неверные данные журнала квот; HTTP не отправлен")
        return state

    @contextmanager
    def _locked_state(self):
        with self._lock:
            if self.path is None:
                yield self._state
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    try:
                        yield self._read()
                    finally:
                        fcntl.flock(lock_file, fcntl.LOCK_UN)

    @contextmanager
    def reserve(self, limits, *, base_url, api_key):
        """Списать квоту после успешной подготовки, непосредственно до POST.

        Исключение внутри with (например, ошибка записи диагностики) отменяет
        резервирование. Ошибки самого HTTP квоту уже расходуют.
        """
        if limits.rpm is None and limits.rpd is None:
            yield
            return
        key = fingerprint({"provider": limits.provider, "base_url": base_url, "api_key": api_key})
        with self._locked_state() as state:
            now = self.clock()
            recent = sorted(t for t in state["providers"].get(key, {}).get("requests", []) if t > now - 86400)
            exhausted, delays = [], []
            for name, budget, window in (("RPM", limits.rpm, 60), ("RPD", limits.rpd, 86400)):
                active = [t for t in recent if t > now - window]
                if budget is not None and len(active) >= budget:
                    exhausted.append(name)
                    delays.append(active[len(active) - budget] + window - now)
            if exhausted:
                raise QuotaExceeded(limits.provider, exhausted, max(delays))
            yield
            state["providers"][key] = {"provider": limits.provider, "requests": [*recent, self.clock()]}
            if self.path is not None:
                atomic_json(self.path, state)
