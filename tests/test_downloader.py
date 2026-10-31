"""Локальные исходники проверяются до использования и до замены файла."""

import pytest
import requests

import downloader


def validate(data):
    if data != b"valid source":
        raise ValueError("повреждённый исходник")
    return data.decode()


def test_valid_local_source_never_downloads_or_rewrites(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"valid source")
    before = source.stat().st_mtime_ns
    monkeypatch.setattr(downloader, "download_bytes", lambda url: pytest.fail("Запрос запрещён"))

    assert downloader.load_source("https://example.org/source", source, validate) == "valid source"
    assert source.stat().st_mtime_ns == before


@pytest.mark.parametrize("local", [None, b"", b"broken source"])
def test_unavailable_source_and_network_have_clear_error(tmp_path, monkeypatch, local):
    source = tmp_path / "source"
    if local is not None:
        source.write_bytes(local)

    def offline(url):
        raise requests.ConnectionError("сеть недоступна")

    monkeypatch.setattr(downloader, "download_bytes", offline)
    with pytest.raises(RuntimeError, match="Нет подходящего локального исходника.*не удалось загрузить"):
        downloader.load_source("https://example.org/source", source, validate)
    assert source.read_bytes() == local if local is not None else not source.exists()


@pytest.mark.parametrize("downloaded", [b"", b"<html>Service unavailable</html>"])
def test_failed_download_validation_preserves_existing_file(tmp_path, monkeypatch, downloaded):
    source = tmp_path / "source"
    source.write_bytes(b"original invalid cache")
    monkeypatch.setattr(downloader, "download_bytes", lambda url: downloaded)
    with pytest.raises(ValueError, match="Непригодный исходник"):
        downloader.load_source("https://example.org/source", source, validate)
    assert source.read_bytes() == b"original invalid cache"
    assert list(tmp_path.iterdir()) == [source]


def test_download_is_saved_after_validation_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    source = tmp_path / "nested" / "source"
    monkeypatch.setattr(downloader, "download_bytes", lambda url: b"valid source")
    assert downloader.load_source("https://example.org/source", source, validate) == "valid source"
    assert source.read_bytes() == b"valid source"
    assert list(source.parent.iterdir()) == [source]
