"""Идентичность одного варианта и его пути без сетевых и файловых эффектов.

Контекст передаётся этапам явно. Год, код, query и суффикс исходного URL
различают версии; вычисление путей само по себе не создаёт каталогов.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from html import unescape
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit, urlunsplit


CODE_RE = re.compile(r"^trvar(?P<number>\d+)(?P<suffix>[\w*+.-]*)$", re.ASCII)


def _canonical_url(url):
    """Нормализовать адрес сайта, сохранив путь и query конкретной версии."""
    parts = urlsplit(unescape(url))
    if parts.scheme not in {"http", "https"} or parts.hostname not in {"alexlarin.net", "www.alexlarin.net"}:
        raise ValueError(f"Нужна ссылка на alexlarin.net: {url}")
    if parts.username or parts.password or parts.port not in {None, 80, 443}:
        raise ValueError("Недопустимые параметры адреса источника")
    return urlunsplit(("https", "alexlarin.net", parts.path, parts.query, ""))


@dataclass
class VariantContext:
    """Идентичность, источники и пути одного варианта без общих изменяемых данных.

    Ключ различает год, исходный code и дополнительные версии URL. Свойства
    путей ничего не создают на диске. ``source_urls`` заполняется при разборе
    HTML, ``download_count`` считает фактические HTTP-попытки этого контекста,
    включая ошибки и отдельные Range-запросы.
    """
    variant_key: str
    number: int
    code: str
    year: int
    source_url: str
    base_dir: Path
    source_urls: dict = field(default_factory=dict)
    download_count: int = 0

    @classmethod
    def from_url(cls, url, base_dir):
        """Проверить ссылку варианта и вычислить его устойчивый ключ и пути."""
        url = _canonical_url(url)
        path = Path(unquote(urlsplit(url).path))
        match = CODE_RE.fullmatch(path.stem)
        if path.suffix.lower() != ".html" or not match or not path.parent.name.isdigit():
            raise ValueError(f"Ссылка не описывает вариант ЕГЭ: {url}")
        year = int(path.parent.name)
        if not 2000 <= year <= 2100:
            raise ValueError(f"Некорректный год варианта: {year}")
        # ~ не допускается в исходном code: экранирование не сталкивается
        # с реально существующими суффиксами вроде _star или _plus.
        safe_code = path.stem.replace("*", "~2a").replace("+", "~2b")
        key = f"ege_{year}_{safe_code}"
        if urlsplit(url).query:
            key += "~q" + sha256(urlsplit(url).query.encode()).hexdigest()[:12]
        return cls(key, int(match["number"]), path.stem, year, url, Path(base_dir))

    @property
    def variant_dir(self):
        """Каталог исходников, кэшей и проверочных данных одного варианта."""
        return self.base_dir / "input" / "variants" / self.variant_key

    @property
    def source_dir(self):
        """Сохранённые HTML/PDF/JS и опубликованные изображения ответов."""
        return self.variant_dir / "source"

    @property
    def cache_dir(self):
        """Проверенные кэши OCR и классификации текущего варианта."""
        return self.variant_dir / "cache"

    @property
    def debug_dir(self):
        """Ответы API, ошибки и данные для проверки вне итоговой выдачи."""
        return self.variant_dir / "debug"

    @property
    def output_dir(self):
        """Итоговая книга Excel и только относящиеся к варианту рисунки."""
        return self.base_dir / "output" / self.variant_key

    @property
    def archive_path(self):
        """Отдельный итоговый ZIP в общем каталоге archives."""
        return self.base_dir / "archives" / f"{self.variant_key}_result.zip"

    @property
    def pdf_path(self):
        """Локальный PDF, соответствующий исходному коду варианта."""
        return self.source_dir / f"{self.code}.pdf"

    @property
    def html_path(self):
        """Локальная страница с проверяемыми ссылками и ответами варианта."""
        return self.source_dir / f"{self.code}.html"
