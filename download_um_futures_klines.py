#!/usr/bin/env python3
"""
Binance USD-M Futures historical downloader + validator
=======================================================

Primary source: data.binance.vision monthly ZIP klines
Fallback source: /fapi/v1/klines REST (for missing ranges)

Examples
--------
python download_um_futures_klines.py --symbols BTCUSDT,ETHUSDT --intervals 15m,1h --start 2023-02-17 --end 2026-02-17 --out data/um_futures

python download_um_futures_klines.py --top 5 --intervals 15m,1h --start 2023-02-17 --end 2026-02-17 --out data/um_futures

python download_um_futures_klines.py --symbols BTCUSDT --intervals 15m --out data/um_futures --validate-only
python download_um_futures_klines.py --symbols BTCUSDT --intervals 15m --out data/um_futures --merge-only
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

from um_futures_data import (
    DEFAULT_BASE_URL,
    fetch_top_symbols,
    parse_date_utc,
    run_download,
    run_validate_or_merge_only,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and validate Binance UM futures klines")
    p.add_argument("--symbols", type=str, help="Comma-separated explicit symbols, e.g. BTCUSDT,ETHUSDT")
    p.add_argument("--top", type=int, help="Top N symbols by 24h quote volume")
    p.add_argument("--intervals", type=str, default="15m,1h")
    p.add_argument("--start", type=str, default="2023-02-17")
    p.add_argument("--end", type=str, default="2026-02-17")
    p.add_argument("--out", type=str, default="data/um_futures")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--rest-fill", type=str, default="true")
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    return p.parse_args()


def parse_symbols(args: argparse.Namespace) -> list[str]:
    explicit = [x.strip().upper() for x in (args.symbols or "").split(",") if x.strip()]
    if explicit:
        return explicit
    if args.top:
        return fetch_top_symbols(args.base_url, args.top)
    raise ValueError("Either --symbols or --top must be provided")


def parse_intervals(raw: str) -> list[str]:
    allowed = {"15m", "1h"}
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one interval required")
    for itv in vals:
        if itv not in allowed:
            raise ValueError(f"Unsupported interval: {itv}")
    return vals


def parse_bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        args = parse_args()
        symbols = parse_symbols(args)
        intervals = parse_intervals(args.intervals)
        start = parse_date_utc(args.start)
        end = parse_date_utc(args.end)
        if end < start:
            raise ValueError("--end must be >= --start")

        # inclusive end date -> end of day UTC
        end = end.replace(hour=23, minute=59, second=59, microsecond=999000)

        if args.validate_only and args.merge_only:
            raise ValueError("--validate-only and --merge-only are mutually exclusive")

        if args.validate_only:
            run_validate_or_merge_only(symbols, intervals, args.out, mode="validate")
        elif args.merge_only:
            run_validate_or_merge_only(symbols, intervals, args.out, mode="merge")
        else:
            run_download(
                symbols=symbols,
                intervals=intervals,
                start=start,
                end=end,
                out_dir=args.out,
                max_workers=args.max_workers,
                rest_fill=parse_bool(args.rest_fill),
                base_url=args.base_url,
            )

        logging.info("Done. symbols=%s intervals=%s", symbols, intervals)
        return 0
    except Exception as exc:
        logging.error("Failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
