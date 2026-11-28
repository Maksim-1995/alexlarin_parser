"""Однократный HTTP через curl; ключ и тело передаются только через stdin."""

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from requests.adapters import BaseAdapter


class CurlAdapter(BaseAdapter):
    """Адаптер requests без повторов и перенаправлений на уровне curl."""

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        connect, read = timeout if isinstance(timeout, tuple) else (timeout, timeout)
        total = (connect or 15) + (read or 120)
        with TemporaryDirectory(prefix="alexparser-curl-") as directory:
            body_path = Path(directory) / "body"
            headers_path = Path(directory) / "headers"
            body = request.body or ""
            if isinstance(body, bytes):
                body = body.decode("utf-8")
            options = {
                "url": request.url, "request": request.method,
                "output": str(body_path), "dump-header": str(headers_path),
                "data-binary": body,
            }
            lines = [f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in options.items()]
            lines.extend("header = " + json.dumps(f"{key}: {value}", ensure_ascii=False)
                         for key, value in request.headers.items()
                         if key.lower() not in {"accept-encoding", "user-agent"})
            lines.append('header = "Accept-Encoding: identity"')
            lines.extend(["silent", "show-error", "retry = 0",
                          f"connect-timeout = {connect or 15}", f"max-time = {total}",
                          'write-out = "%{http_code}"'])
            try:
                result = subprocess.run(
                    ["curl", "--disable", "--config", "-"], input="\n".join(lines) + "\n",
                    text=True, capture_output=True, timeout=total + 5,
                )
            except subprocess.TimeoutExpired:
                raise requests.Timeout("Groq curl: истёк срок запроса") from None
            except OSError:
                raise requests.ConnectionError("Groq curl: не удалось запустить curl") from None
            if result.returncode == 28:
                raise requests.Timeout("Groq curl: истёк срок запроса")
            if result.returncode or not result.stdout.strip().isdigit():
                # stderr не публикуется: он может содержать данные запроса.
                raise requests.ConnectionError("Groq curl: ошибка транспорта")
            response = requests.Response()
            response.status_code = int(result.stdout.strip())
            response.url, response.request = request.url, request
            response._content = body_path.read_bytes()
            response.encoding = "utf-8"
            for line in headers_path.read_text(encoding="iso-8859-1").splitlines():
                if line.startswith("HTTP/"):
                    response.headers.clear()
                elif ":" in line:
                    key, value = line.split(":", 1)
                    response.headers[key.strip()] = value.strip()
            return response

    def close(self):
        pass
