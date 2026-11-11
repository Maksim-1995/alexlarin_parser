"""Точка входа одиночной и пакетной обработки через прямой DeepSeek.

Общий поток и отчёты находятся в pipeline.py. Прежние валидаторы 542
повторно экспортируются из legacy_sources.py для совместимости импортов.
"""

import argparse
import json
import sys
from pathlib import Path

# Старые публичные имена остаются доступны как main.parse_source_page и т. п.
# Ограничения формата 542 сосредоточены в отдельном legacy-модуле.
from legacy_sources import (
    parse_metadata,
    parse_source_page,
    read_extra_answers,
    read_js_source,
    read_pdf_source,
    validate_answer_image,
)
from parsers import parse_js_answers, parse_tasks


DEFAULT_PAGE_URL = "https://alexlarin.net/ege/2026/trvar542.html"
BASE_DIR = Path(__file__).resolve().parent
# Сохраняется для прежних импортов и локальных тестовых окружений.
INPUT_DIR = BASE_DIR / "input"


def main(page_url=DEFAULT_PAGE_URL, *, fresh_ai=False, local_only=False):
    """Обработать один вариант через общий изолированный поток.

    Args:
        page_url: URL страницы варианта; прежний адрес 542 остаётся значением
            по умолчанию для совместимости одиночного запуска.
        fresh_ai: Использовать новый изолированный AI-кэш для измерений.
        local_only: Запретить сетевое скачивание исходников; AI-кэш и
            разрешённые модельные запросы работают по обычным правилам.

    Returns:
        Отчёт run_batch со статусом, сведениями о варианте и путём манифеста.
    """
    from pipeline import run_batch
    from variant_sources import VariantContext
    return run_batch(BASE_DIR, [VariantContext.from_url(page_url, BASE_DIR)],
                     fresh_ai=fresh_ai, local_only=local_only)


def build_argument_parser():
    """Создать parser CLI без запуска обработки или обращения к источникам.

    Returns:
        ArgumentParser с выбором одиночного, пакетного и возобновляемого
        запуска, режимом планирования и предохранителем размера пакета.
    """
    parser = argparse.ArgumentParser(description="Изолированный парсер вариантов alexlarin.net")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--url", help="Страница одного варианта; по умолчанию прежний №542")
    choice.add_argument("--batch", help="Основные номера из реальных href архива: 1-5 либо 1,2,3")
    choice.add_argument("--resume", type=Path, help="Манифест предыдущего запуска reports/<run_id>/manifest.json")
    parser.add_argument("--plan", action="store_true", help="Показать выбор и пути без платных вызовов")
    parser.add_argument("--fresh-ai", action="store_true", help="Изолированные новые AI-кэши, прежние сохраняются")
    parser.add_argument("--local-only", action="store_true", help="Использовать только проверенные локальные исходники")
    parser.add_argument("--max-variants", type=int, default=5, help="Предохранитель размера платного пакета (по умолчанию 5)")
    return parser


def cli(argv=None):
    """Выполнить CLI, сохранив план без AI и лимит платного пакета.

    Args:
        argv: Список аргументов без имени программы; None читает sys.argv.

    Returns:
        Код завершения: 0 для SUCCESS или плана, 2 для NEEDS_REVIEW,
        1 при ошибке и 130 при прерывании пользователем.

    Raises:
        SystemExit: argparse обрабатывает --help и ошибки синтаксиса CLI.
    """
    from pipeline import parse_selection, plan_run, run_batch, selected_contexts
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        numbers = parse_selection(args.batch) if args.batch else None
        contexts = selected_contexts(BASE_DIR, numbers=numbers,
                                     url=args.url or (DEFAULT_PAGE_URL if not args.batch and not args.resume else None),
                                     resume=args.resume, local_only=args.local_only)
        if args.plan:
            print(json.dumps(plan_run(BASE_DIR, contexts), ensure_ascii=False, indent=2))
            return 0
        if args.max_variants <= 0 or len(contexts) > args.max_variants:
            raise ValueError("Размер пакета превышает --max-variants; сначала проверьте --plan")
        result = run_batch(BASE_DIR, contexts, fresh_ai=args.fresh_ai,
                           local_only=args.local_only, resume_from=args.resume)
        return {"SUCCESS": 0, "NEEDS_REVIEW": 2, "FAILED": 1}[result["status"]]
    except KeyboardInterrupt:
        print("Запуск прерван; проверенные страницы и отчёт сохранены", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
