"""JSON-ответ с происхождением, которое не добавляет полей в схему модели."""


class ModelResponse(dict):
    def __init__(self, data, provenance=None):
        super().__init__(data)
        self.provenance = dict(provenance) if isinstance(provenance, dict) else None


def response_provenance(response, configuration):
    provenance = getattr(response, "provenance", None)
    if isinstance(provenance, dict) and isinstance(provenance.get("provider"), str):
        return dict(provenance)
    # В старом прямом протоколе провайдер однозначен. По конфигурации цепочки
    # нельзя установить, кто фактически ответил на сохранённый запрос.
    if not configuration.get("free_providers") and isinstance(configuration.get("provider"), str):
        return {key: configuration[key] for key in ("provider", "model", "base_url") if key in configuration}
    return {"provider": "unknown", "reason": "Фактический провайдер не сохранён в результате"}


def pages_provenance(pages):
    providers = {page["provider"] for page in pages}
    result = {"provider": next(iter(providers)) if len(providers) == 1 else "mixed", "pages": pages}
    for key in ("model", "base_url"):
        values = {page.get(key) for page in pages}
        if len(values) == 1 and None not in values:
            result[key] = next(iter(values))
    return result
