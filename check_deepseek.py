"""Проверка подключения к DeepSeek через NVIDIA или напрямую."""

import argparse
import json
import os
import time

import requests


parser = argparse.ArgumentParser()
parser.add_argument(
    "--provider",
    choices=("nvidia", "deepseek"),
    required=True,
)
args = parser.parse_args()

profiles = {
    "nvidia": {
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "key_name": "NVIDIA_API_KEY",
        "model": "deepseek-ai/deepseek-v4.1-flash",
    },
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "key_name": "DEEPSEEK_API_KEY",
        "model": "deepseek-flash",
    },
}

profile = profiles[args.provider]
api_key = os.environ.get(profile["key_name"])

if not api_key:
    raise SystemExit(
        f"Не задана переменная {profile['key_name']}. "
        "Проверь .env и загрузку переменных."
    )

payload = {
    "model": profile["model"],
    "messages": [
        {
            "role": "user",
            "content": "Ответь ровно одним словом: Работает",
        }
    ],
    "max_tokens": 2048,
    "stream": False,
}

# Настройка прямого API DeepSeek.
# Параметры разных провайдеров задаём отдельно.
if args.provider == "deepseek":
    payload["thinking"] = {"type": "disabled"}

print("Провайдер:", args.provider)
print("Модель:", profile["model"])
print("Отправляем один запрос...", flush=True)

started = time.monotonic()

try:
    response = requests.post(
        profile["url"],
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        json=payload,
        timeout=(15, 120),
    )
except requests.RequestException as error:
    print("Секунд:", round(time.monotonic() - started, 1))
    raise SystemExit(
        f"Ошибка подключения: {type(error).__name__}: {error}"
    )

print("HTTP:", response.status_code)
print("Секунд:", round(time.monotonic() - started, 1))

try:
    data = response.json()
except ValueError:
    raise SystemExit(response.text[:2000])

if not response.ok or not data.get("choices"):
    raise SystemExit(json.dumps(data, ensure_ascii=False, indent=2))

choice = data["choices"][0]

print("Причина завершения:", choice.get("finish_reason"))
print("Ответ:", choice.get("message", {}).get("content"))
print(
    "Расход токенов:",
    json.dumps(data.get("usage"), ensure_ascii=False),
)