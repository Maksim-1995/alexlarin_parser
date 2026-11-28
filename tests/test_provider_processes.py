"""Квоты и cooldown между независимыми процессами; HTTP полностью подставной."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import format_datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import requests

from alexlarin_parser import provider, rate_limits
from alexlarin_parser.paths import PROJECT_ROOT


PROVIDERS = ("gemini", "nvidia-nim", "groq", "openrouter", "deepseek-direct")
TEST_KEY = "process-test-not-a-real-key"


def _worker(directory, ledger, name, stage, limit_kind, scenario, worker_id, now):
    """Новый интерпретатор: своя сессия/память, общий только журнал на диске."""
    directory, ledger, now = Path(directory), Path(ledger), float(now)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(now, tz=tz)

    provider.datetime = FrozenDatetime
    real_atomic_json = rate_limits.atomic_json

    def delayed_write(path, data):
        # Медленная запись расширяет окно гонки read/check/write. flock
        # должен защищать весь интервал, а не только замену JSON.
        if limit_kind != "none" and Path(path) == ledger:
            time.sleep(0.15)
        real_atomic_json(path, data)

    rate_limits.atomic_json = delayed_write

    class Session:
        def __init__(self):
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            if scenario == "timeout":
                raise requests.ReadTimeout("подставной тайм-аут")
            response = requests.Response()
            response.status_code = {"429": 429, "503-date": 503}.get(scenario, 200)
            response.headers["Retry-After"] = (
                format_datetime(datetime.fromtimestamp(now + 37, timezone.utc), usegmt=True)
                if scenario == "503-date" else "37")
            response._content = json.dumps({"choices": [{"finish_reason": "stop",
                                                         "message": {"content": '{"tasks": []}'}}]}).encode()
            return response

        def close(self):
            pass

    limits = () if limit_kind == "none" else (provider.ProviderLimits(name, **{limit_kind: 1}),)
    settings = provider.DeepSeekSettings(
        api_key=TEST_KEY if name == "deepseek-direct" else "",
        free_providers=() if name == "deepseek-direct" else (
            provider._FreeProvider(name, "https://test.invalid/v1", TEST_KEY, "test:free", "test:free"),),
        rate_limits=limits, rate_limit_path=ledger,
    )
    session = Session()
    client = provider.DeepSeekClient(settings, session=session,
                                     rate_limiter=provider.RequestRateLimiter(ledger, clock=lambda: now))
    (directory / f"ready-{worker_id}").touch()
    deadline = time.monotonic() + 10
    while not (directory / "start").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Родитель не разрешил старт проверки")
        time.sleep(0.01)
    try:
        result = client.request_json([{"role": "user", "content": "Верни JSON"}],
                                     stage=stage, debug_dir=directory / f"debug-{worker_id}")
        status, error, provenance = "ok", None, result.provenance
    except provider.ProviderError as failure:
        status, error, provenance = "blocked", str(failure), None
    finally:
        client.close()
    print(json.dumps({"pid": os.getpid(), "status": status, "error": error, "provenance": provenance,
                      "http_calls": session.calls, "ocr_calls": client.counter.ocr,
                      "classification_calls": client.counter.classification,
                      "attempts": client.attempts, "diagnostics": client.provider_diagnostics}))


def run_processes(directory, ledger, name, limit_kind, scenarios, now=100000):
    directory.mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT), env.get("PYTHONPATH", "")))
    processes = []
    try:
        for index, scenario in enumerate(scenarios):
            command = [sys.executable, str(Path(__file__).resolve()), str(directory), str(ledger), name,
                       "ocr" if index == 0 else "classification", limit_kind, scenario, str(index), str(now)]
            processes.append(subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, text=True,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        deadline = time.monotonic() + 10
        while len(list(directory.glob("ready-*"))) != len(processes):
            if any(process.poll() is not None for process in processes):
                raise AssertionError("Дочерний процесс завершился до старта")
            if time.monotonic() >= deadline:
                raise AssertionError("Процессы не подготовились к проверке")
            time.sleep(0.01)
        (directory / "start").touch()
        with ThreadPoolExecutor(max_workers=len(processes)) as executor:
            outputs = list(executor.map(lambda process: process.communicate(timeout=10), processes))
        for process, (_, stderr) in zip(processes, outputs):
            assert process.returncode == 0, stderr
        return [json.loads(stdout) for stdout, _ in outputs]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("limit_kind", ["rpm", "rpd"])
def test_shared_quota_is_atomic_between_two_processes_and_stages(tmp_path, name, limit_kind):
    ledger = tmp_path / "quotas.json"
    results = run_processes(tmp_path / "workers", ledger, name, limit_kind, ["ok", "ok"])
    assert len({result["pid"] for result in results}) == 2
    assert sorted(result["status"] for result in results) == ["blocked", "ok"]
    assert sum(result["http_calls"] for result in results) == 1
    assert sum(result["ocr_calls"] + result["classification_calls"] for result in results) == 1
    assert sum(len(result["attempts"]) for result in results) == 1
    winner = next(result for result in results if result["status"] == "ok")
    loser = next(result for result in results if result["status"] == "blocked")
    assert winner["provenance"]["provider"] == name and not loser["attempts"]
    skipped = next(event for event in loser["diagnostics"] if event["provider"] == name)
    assert skipped["kind"] == "quota" and limit_kind.upper() in skipped["reason"]
    record, = json.loads(ledger.read_text())["providers"].values()
    assert record["requests"] == [100000] and TEST_KEY not in ledger.read_text()


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("scenario,delay", [("429", 37), ("503-date", 37), ("timeout", 60)])
def test_cooldown_survives_new_process_without_local_quotas(tmp_path, name, scenario, delay):
    ledger = tmp_path / "quotas.json"
    first, = run_processes(tmp_path / "first", ledger, name, "none", [scenario])
    assert first["status"] == "blocked" and first["http_calls"] == 1
    record, = json.loads(ledger.read_text())["providers"].values()
    assert record["cooldown_until"] == 100000 + delay
    second, = run_processes(tmp_path / "second", ledger, name, "none", ["ok"], now=100000 + delay - 1)
    assert second["pid"] != first["pid"] and second["http_calls"] == 0 and not second["attempts"]
    skipped = next(event for event in second["diagnostics"] if event["provider"] == name)
    assert skipped["kind"] == "cooldown" and "1 с" in skipped["reason"]
    third, = run_processes(tmp_path / "third", ledger, name, "none", ["ok"], now=100000 + delay)
    assert third["status"] == "ok" and third["http_calls"] == 1
    assert third["provenance"]["provider"] == name


if __name__ == "__main__":
    _worker(*sys.argv[1:])
