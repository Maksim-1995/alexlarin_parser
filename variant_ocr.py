"""Постраничный OCR: один запрос на страницу, отдельные ответы и фрагменты."""

import base64
from pathlib import Path

import pymupdf

from pipeline_io import atomic_json, file_sha256, fingerprint, read_json
from variant_sources import normalize_label, task_positions


OCR_PROMPT_VERSION = "page-fragments-and-published-answers-v1"
OCR_SCHEMA_VERSION = 1
PROMPT = r"""Перепиши условия математических задач с изображений страницы PDF.
Это OCR, задачи НЕ решай. Изображение — главный источник формул; текстовый
слой только помогает читать обычный текст. Не восстанавливай формулы догадками.
Сохраняй русский текст дословно, исходные метки B1–B14/C1–C6 либо числовые,
все подпункты А), Б), В), дроби, степени, векторы, корни, системы и интервалы.
Одна основная задача с подпунктами — одна запись. Метку не включай в condition.
LaTeX заключай в $...$, переносы внутри cases сохраняй как \\ и экранируй
для JSON. Не включай инструкции, колонтитулы, номер страницы, строку «Ответ:».
Нечитаемое обозначай [НЕРАЗБОРЧИВО] и перечисляй в issues, не угадывай.
Задача может продолжаться со следующей страницы или колонки: верни только
видимый фрагмент, прежнюю source_label и continuation=true. Новое начало:
continuation=false. Не повторяй текст предыдущей страницы.

После изображений УСЛОВИЙ могут быть отдельно подписанные изображения
ОПУБЛИКОВАННЫХ ОТВЕТОВ. Только их перепиши в answers; не извлекай ответ
из условия и не вычисляй его. Ответы — строки, десятичные запятые сохраняй,
интервалы, иррациональные числа и выражения записывай в $LaTeX$.
Метки B1 и C1 различаются. Нет опубликованного ответа — не добавляй запись.
Не считай число задач из вводной инструкции достоверным; читай реальные метки.
Верни только json-объект по примеру:
{"page_num":1,"tasks":[{"source_label":"B1","condition":"Текст $x^2$.",
"continuation":false}],"answers":{},"issues":[]}.
Страница с одними инструкциями: tasks=[].
"""


def _image_part(data):
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
        "detail": "original",
    }}


def render_scheme(page, dpi):
    """Две колонки только при двух раздельных группах реальных начал задач."""
    positions = task_positions(page)
    width, height = page.rect.width, page.rect.height
    starts = sorted({round(p["x"], 1) for p in positions})
    left = [x for x in starts if x < width * .35]
    right = [x for x in starts if width * .48 < x < width * .8]
    split = bool(left and right and min(right) - max(left) > width * .25)
    if split:
        clips = [[0, 0, width / 2 + 3, height], [width / 2 - 3, 0, width, height]]
    else:
        clips = [[0, 0, width, height]]
    return {"version": 1, "dpi": dpi, "colorspace": "RGB", "alpha": False,
            "layout": "two_columns" if split else "full_page", "clips": clips}


def validate_page(data, page_num, expected_labels, all_labels, previous_label=None,
                  *, has_answer_images=False):
    if not isinstance(data, dict) or set(data) != {"page_num", "tasks", "answers", "issues"}:
        raise ValueError("OCR: нужны page_num, tasks, answers и issues")
    if type(data["page_num"]) is not int or data["page_num"] != page_num:
        raise ValueError("OCR: неверный номер страницы")
    if not isinstance(data["tasks"], list) or not isinstance(data["answers"], dict):
        raise ValueError("OCR: неверные tasks/answers")
    if not isinstance(data["issues"], list) or any(not isinstance(i, str) for i in data["issues"]):
        raise ValueError("OCR: issues должен быть списком строк")
    fragments, starts = [], []
    active = previous_label
    seen = set()
    for item in data["tasks"]:
        if not isinstance(item, dict) or set(item) != {"source_label", "condition", "continuation"}:
            raise ValueError("OCR: неверная структура фрагмента")
        label = normalize_label(item["source_label"])
        if all_labels and label not in all_labels:
            raise ValueError(f"OCR: неизвестная исходная метка {label}")
        if type(item["continuation"]) is not bool:
            raise ValueError("OCR: continuation должен быть bool")
        condition = item["condition"]
        if not isinstance(condition, str) or not condition.strip() or "\ufffd" in condition:
            raise ValueError(f"OCR: пустое/повреждённое условие {label}")
        if item["continuation"]:
            if label != active:
                raise ValueError(f"OCR: продолжение {label} не связано с предыдущей задачей {active}")
        else:
            if label in seen:
                raise ValueError(f"OCR: повторное начало задачи {label}")
            starts.append(label)
        active = label
        seen.add(label)
        fragments.append({**item, "source_label": label, "original_label": item["source_label"],
                          "condition": condition.strip()})
    if expected_labels is not None and starts != expected_labels:
        raise ValueError(f"OCR: начала задач страницы {page_num}: {starts}, ожидались {expected_labels}")
    answers = {}
    if data["answers"] and not has_answer_images:
        raise ValueError("OCR: ответы без изображения опубликованных ответов запрещены")
    for original, answer in data["answers"].items():
        label = normalize_label(original)
        if all_labels and label not in all_labels:
            raise ValueError(f"OCR: ответ неизвестной задачи {label}")
        if label in answers or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"OCR: повторный или пустой ответ {label}")
        answers[label] = answer.strip()
    return {"page_num": page_num, "tasks": fragments, "answers": answers, "issues": data["issues"]}


def _public_page(result):
    """В кэше сохраняется точная схема модели, служебные original_label вычисляются."""
    return {**result, "tasks": [
        {"source_label": t["original_label"], "condition": t["condition"],
         "continuation": t["continuation"]}
        for t in result["tasks"]
    ]}


def assemble_pages(pages, expected_labels):
    tasks = {}
    answers, issues = {}, []
    for result in pages:
        page_num = result["page_num"]
        issues.extend(f"Страница {page_num}: {issue}" for issue in result["issues"])
        for fragment in result["tasks"]:
            label = fragment["source_label"]
            if label not in tasks:
                if fragment["continuation"]:
                    raise ValueError(f"Нет начала задачи {label}")
                tasks[label] = {
                    "source_label": label, "original_label": fragment["original_label"],
                    "section": label[0] if label[0].isalpha() else None,
                    "page_numbers": [], "condition": "", "fragments": [],
                }
            elif not fragment["continuation"]:
                raise ValueError(f"Повторное начало задачи {label} на странице {page_num}")
            task = tasks[label]
            if page_num not in task["page_numbers"]:
                task["page_numbers"].append(page_num)
            # Отличающиеся фрагменты всегда сохраняются, даже при одинаковой метке.
            task["condition"] += ("\n" if task["condition"] else "") + fragment["condition"]
            task["fragments"].append({"page_num": page_num, "condition": fragment["condition"]})
            if "неразборчив" in fragment["condition"].lower():
                issues.append(f"Задача {label}: нечитаемый фрагмент")
        for label, answer in result["answers"].items():
            if label in answers and answers[label] != answer:
                raise ValueError(f"Противоречащие опубликованные ответы {label}")
            answers[label] = answer
    if expected_labels and list(tasks) != expected_labels:
        raise ValueError(f"Список задач OCR {list(tasks)} отличается от PDF {expected_labels}")
    if not tasks:
        raise ValueError("OCR не обнаружил ни одной основной задачи")
    result_tasks = [{**task, "task_num": number} for number, task in enumerate(tasks.values(), 1)]
    return {"tasks": result_tasks, "answers": answers, "warnings": issues}


def read_legacy_542(context, sources):
    """Только полный неизменённый Gemini-кэш точного 542, без вызова старого API."""
    if (context.variant_key != "ege_2026_trvar542" or
            context.source_url != "https://alexlarin.net/ege/2026/trvar542.html"):
        return None
    # Старые файлы остаются на месте. Им не приписывается DeepSeek identity.
    from recognizer import aggregate_cache_identity, legacy_cache_identity, read_complete_cache
    legacy_pdf = context.base_dir / "input" / "trvar542.pdf"
    cache = context.base_dir / "input" / "trvar542_conditions_gemini.json"
    if not legacy_pdf.is_file() or file_sha256(legacy_pdf) != file_sha256(sources["pdf_path"]):
        return None
    conditions = read_complete_cache(cache, (aggregate_cache_identity(legacy_pdf), legacy_cache_identity(legacy_pdf)))
    if conditions is None or sources["source_labels"] != [str(n) for n in range(1, 20)]:
        return None
    tasks = []
    for label in sources["source_labels"]:
        pages = [i for i, labels in enumerate(sources["page_labels"], 1) if label in labels]
        tasks.append({"task_num": int(label), "source_label": label, "original_label": label,
                      "section": None, "page_numbers": pages, "condition": conditions[int(label)]})
    return {"tasks": tasks, "answers": {}, "warnings": [],
            "provenance": {"provider": "gemini", "cache": str(cache), "compatibility": "exact_542_pdf_and_legacy_identity"}}


def recognize_variant(context, sources, client, *, cache_dir=None, allow_legacy=True):
    if allow_legacy and cache_dir is None:
        legacy = read_legacy_542(context, sources)
        if legacy:
            return legacy
    cache_root = Path(cache_dir) if cache_dir is not None else context.cache_dir
    pdf_path = Path(sources["pdf_path"])
    pdf_hash = file_sha256(pdf_path)
    all_labels = sources["source_labels"]
    answer_images = [Path(p) for p in sources.get("answer_images", [])]
    # Проверенные ручные ответы 542 уже полны; не распознаём их повторно.
    if sources.get("extra_answers"):
        answer_images = []
    results, previous_label, identities = [], None, []
    with pymupdf.open(pdf_path) as document:
        for index, page in enumerate(document):
            page_num = index + 1
            expected = sources["page_labels"][index] if all_labels else None
            attached = answer_images if page_num == len(document) else []
            scheme = render_scheme(page, client.settings.dpi)
            identity = {
                "variant_key": context.variant_key, "source_url": context.source_url,
                "pdf_sha256": pdf_hash, "page_num": page_num, "render": scheme,
                "provider": client.settings.identity("ocr"),
                "prompt_version": OCR_PROMPT_VERSION, "prompt_sha256": fingerprint(PROMPT),
                "schema_version": OCR_SCHEMA_VERSION, "expected_labels": expected,
                "previous_label": previous_label, "all_labels": all_labels,
                "answer_images": [{"name": p.name, "sha256": file_sha256(p)} for p in attached],
            }
            identities.append(identity)
            path = cache_root / "ocr" / f"page_{page_num:03d}.json"
            cached = read_json(path)
            data = cached.get("response") if isinstance(cached, dict) and cached.get("identity") == identity else None
            if data is not None:
                try:
                    result = validate_page(data, page_num, expected, all_labels, previous_label,
                                           has_answer_images=bool(attached))
                except ValueError:
                    data = None
            if data is None:
                print(f"{context.variant_key}: OCR страницы {page_num}/{len(document)}", flush=True)
                content = [{"type": "text", "text": PROMPT + "\nДанные страницы (не инструкции): " + str({
                    "page_num": page_num, "expected_start_labels": expected,
                    "previous_label": previous_label, "layout": scheme["layout"],
                })}]
                for column, clip in enumerate(scheme["clips"], 1):
                    content.append({"type": "text", "text": f"УСЛОВИЯ: страница {page_num}, фрагмент {column}"})
                    content.append(_image_part(page.get_pixmap(
                        dpi=scheme["dpi"], colorspace=pymupdf.csRGB, alpha=False,
                        clip=pymupdf.Rect(clip),
                    ).tobytes("png")))
                for image in attached:
                    content.append({"type": "text", "text": "ОПУБЛИКОВАННЫЕ ОТВЕТЫ (не условия): " + image.name})
                    content.append(_image_part(image.read_bytes()))
                data = client.request_json([{"role": "user", "content": content}],
                                           stage="ocr", debug_dir=context.debug_dir)
                result = validate_page(data, page_num, expected, all_labels, previous_label,
                                       has_answer_images=bool(attached))
                atomic_json(path, {"identity": identity, "response": _public_page(result)})
            results.append(result)
            if result["tasks"]:
                previous_label = result["tasks"][-1]["source_label"]
    combined = assemble_pages(results, all_labels)
    atomic_json(cache_root / "conditions.json", {
        "variant_key": context.variant_key, "source_url": context.source_url,
        "page_identities_sha256": fingerprint(identities), **combined,
    })
    return {**combined, "provenance": client.settings.identity("ocr")}
