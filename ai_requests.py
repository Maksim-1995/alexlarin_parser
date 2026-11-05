"""Фактические вызовы SDK в пределах одного запуска пайплайна."""

from dataclasses import dataclass


@dataclass
class AIRequestCounter:
    """Счётчик увеличивается непосредственно перед generate_content."""

    ocr: int = 0
    classification: int = 0

    @property
    def total(self):
        return self.ocr + self.classification
