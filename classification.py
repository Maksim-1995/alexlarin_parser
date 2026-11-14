"""Совместимые импорты классификации из пакета alexlarin_parser.

Логика находится в classification.catalogues, classification.service и
classification.legacy внутри пакета. Фасад сохраняет имена функций и
констант; изменения настроек/monkeypatch адресуются их реальному владельцу.
"""

from alexlarin_parser.classification import (
    TASK_NUMBERS,
    Catalog,
    normalize_catalog_id,
    load_catalog,
    allowed_categories,
    build_category_choices,
    normalize_topic_name,
    resolve_new_topic,
    decode_category_choices,
    validate_classification,
    MODEL,
    THINKING_LEVEL,
    CACHE_VERSION,
    RULES_VERSION,
    REQUEST_TIMEOUT_MS,
    MAX_OUTPUT_TOKENS,
    PROMPT,
    SCHEMA,
    prepare_tasks,
    request_gemini,
    classify_tasks,
    VARIANT_CACHE_VERSION,
    VARIANT_RULES_VERSION,
    VARIANT_PROMPT_VERSION,
    VARIANT_PROMPT,
    VARIANT_SCHEMA,
    NUMERIC_VARIANT_PROMPT,
    NUMERIC_VARIANT_PROMPT_VERSION,
    NUMERIC_VARIANT_SCHEMA,
    prepare_variant_tasks,
    classify_variant,
    classification_review_warnings,
)
from alexlarin_parser.classification import __all__
