"""Число фактических модельных попыток по этапам одного клиента/запуска."""

from dataclasses import dataclass


@dataclass
class AIRequestCounter:
    """OCR и классификация считаются отдельно перед обращением к транспорту.

    Прямой DeepSeek-клиент увеличивает счётчик перед HTTP POST; старый
    совместимый транспорт — перед вызовом SDK. Ошибки и тайм-ауты тоже
    считаются попытками. Чтение кэша и загрузка исходников сюда не входят.
    """

    ocr: int = 0
    classification: int = 0

    @property
    def total(self):
        """Вернуть общее число OCR- и классификационных попыток."""
        return self.ocr + self.classification
