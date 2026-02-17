#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

BINANCE_DATA_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
DEFAULT_BASE_URL = "https://fapi.binance.com"
MAX_RETRIES = 5
LIMIT = 1500

COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

STEP_MS = {"15m": 900000, "1h": 3600000}


@dataclass
class ValidationReport:
    total_rows: int
    duplicates_removed: int
    gap_count: int
    gap_ranges: list[dict[str, int]]
    first_ts: int | None
    last_ts: int | None


# ---------- HTTP helpers ----------
def _http_get(url: str, timeout: int = 30) -> bytes:
    if requests is not None:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code == 404:
                    raise FileNotFoundError(url)
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {url}")
                return resp.content
            except Exception:
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(min(2 ** attempt, 16) + random.random())
        raise RuntimeError("unreachable")

    req = Request(url)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urlopen(req, timeout=timeout) as r:  # nosec B310
                return r.read()
        except HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            if attempt == MAX_RETRIES:
                raise
        except URLError:
            if attempt == MAX_RETRIES:
                raise
        time.sleep(min(2 ** attempt, 16) + random.random())
    raise RuntimeError("unreachable")


def _rest_get(base_url: str, endpoint: str, params: dict[str, str | int]) -> list:
    url = f"{base_url}{endpoint}?{urlencode(params)}"
    data = _http_get(url, timeout=30)
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected payload from {url}")
    return payload


# ---------- date helpers ----------
def parse_date_utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)


def month_iter(start: datetime, end: datetime) -> Iterable[tuple[int, int]]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def month_start_end_ms(year: int, month: int) -> tuple[int, int]:
    start = datetime(year, month, 1, tzinfo=UTC)
    if month == 12:
        nxt = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        nxt = datetime(year, month + 1, 1, tzinfo=UTC)
    return int(start.timestamp() * 1000), int(nxt.timestamp() * 1000) - 1


# ---------- checksum ----------
def parse_checksum_text(txt: str) -> str | None:
    # typical format: "<sha256>  <filename>"
    parts = txt.strip().split()
    if not parts:
        return None
    cand = parts[0].strip()
    if len(cand) == 64:
        return cand.lower()
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- CSV ----------
def write_rows_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)


def read_rows_csv(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            return rows
        for row in r:
            if len(row) >= 12:
                rows.append(row[:12])
    return rows


# ---------- validation ----------
def validate_rows(rows: list[list[str]], interval: str) -> tuple[list[list[str]], ValidationReport]:
    step = STEP_MS[interval]
    cleaned = sorted(rows, key=lambda x: int(x[0]))

    deduped: list[list[str]] = []
    seen: set[int] = set()
    dups = 0
    for row in cleaned:
        ts = int(row[0])
        if ts in seen:
            dups += 1
            continue
        seen.add(ts)
        # OHLC consistency
        o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        if h < max(o, c) or l > min(o, c):
            continue
        deduped.append(row)

    gaps: list[dict[str, int]] = []
    for i in range(1, len(deduped)):
        prev_ts = int(deduped[i - 1][0])
        cur_ts = int(deduped[i][0])
        delta = cur_ts - prev_ts
        if delta != step:
            if delta > step:
                gaps.append({"start": prev_ts + step, "end": cur_ts - step})
            # if delta < step it's disorder/dup already handled

    report = ValidationReport(
        total_rows=len(deduped),
        duplicates_removed=dups,
        gap_count=len(gaps),
        gap_ranges=gaps,
        first_ts=int(deduped[0][0]) if deduped else None,
        last_ts=int(deduped[-1][0]) if deduped else None,
    )
    return deduped, report


def validate_csv_file(path: Path, interval: str) -> tuple[bool, ValidationReport]:
    if not path.exists():
        return False, ValidationReport(0, 0, 0, [], None, None)
    rows = read_rows_csv(path)
    cleaned, report = validate_rows(rows, interval)
    if len(cleaned) != len(rows):
        write_rows_csv(path, cleaned)
    return True, report


def write_validation_report(path: Path, report: ValidationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, indent=2)


# ---------- monthly download ----------
def download_month_zip(symbol: str, interval: str, year: int, month: int, out_root: Path) -> Path | None:
    ym = f"{year:04d}-{month:02d}"
    base = f"{BINANCE_DATA_BASE}/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
    raw_path = out_root / "klines" / symbol / interval / "raw" / f"{ym}.csv"

    if raw_path.exists():
        ok, rep = validate_csv_file(raw_path, interval)
        if ok and rep.total_rows > 0:
            logging.info("[SKIP] %s %s %s already valid", symbol, interval, ym)
            return raw_path

    try:
        zip_bytes = _http_get(base)
    except FileNotFoundError:
        logging.warning("[MISS] monthly zip not found: %s", base)
        return None

    checksum_url = base + ".CHECKSUM"
    checksum_ok = None
    try:
        checksum_txt = _http_get(checksum_url).decode("utf-8", errors="ignore")
        expected = parse_checksum_text(checksum_txt)
        if expected:
            got = hashlib.sha256(zip_bytes).hexdigest()
            checksum_ok = got == expected
            if not checksum_ok:
                logging.error("[BAD] checksum mismatch %s %s %s", symbol, interval, ym)
                return None
        else:
            logging.info("[WARN] checksum present but unparsable for %s", ym)
    except FileNotFoundError:
        logging.info("[INFO] checksum missing for %s %s %s", symbol, interval, ym)

    if checksum_ok is True:
        logging.info("[OK] checksum verified %s %s %s", symbol, interval, ym)

    # extract first CSV from zip
    tmp_zip = out_root / "_tmp" / f"{symbol}-{interval}-{ym}.zip"
    tmp_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip.write_bytes(zip_bytes)

    try:
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not names:
                logging.error("[BAD] zip no csv %s", tmp_zip)
                return None
            data = zf.read(names[0]).decode("utf-8", errors="ignore")
    finally:
        try:
            tmp_zip.unlink()
        except Exception:
            pass

    rows: list[list[str]] = []
    reader = csv.reader(data.splitlines())
    for row in reader:
        if len(row) >= 12 and row[0].isdigit():
            rows.append(row[:12])

    cleaned, _ = validate_rows(rows, interval)
    write_rows_csv(raw_path, cleaned)
    logging.info("[DONE] monthly %s %s %s rows=%d", symbol, interval, ym, len(cleaned))
    return raw_path


# ---------- merge ----------
def merge_raw_files(symbol: str, interval: str, out_root: Path, start_ms: int, end_ms: int) -> Path:
    raw_dir = out_root / "klines" / symbol / interval / "raw"
    merged_path = out_root / "klines" / symbol / interval / "merged.csv"
    rows: list[list[str]] = []
    if raw_dir.exists():
        for p in sorted(raw_dir.glob("*.csv")):
            rows.extend(read_rows_csv(p))

    # crop to requested range
    rows = [r for r in rows if start_ms <= int(r[0]) <= end_ms]
    cleaned, _ = validate_rows(rows, interval)
    write_rows_csv(merged_path, cleaned)

    # optional parquet
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(merged_path)
        df.to_parquet(merged_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass

    return merged_path


# ---------- REST fill ----------
def fetch_klines_rest_range(base_url: str, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list[str]]:
    rows: list[list[str]] = []
    cur = start_ms
    step = STEP_MS[interval]
    while cur <= end_ms:
        payload = _rest_get(
            base_url,
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cur,
                "endTime": end_ms,
                "limit": LIMIT,
            },
        )
        if not payload:
            break
        batch = [
            [
                str(x[0]),
                str(x[1]),
                str(x[2]),
                str(x[3]),
                str(x[4]),
                str(x[5]),
                str(x[6]),
                str(x[7]),
                str(x[8]),
                str(x[9]),
                str(x[10]),
                str(x[11]),
            ]
            for x in payload
        ]
        rows.extend(batch)
        last_open = int(batch[-1][0])
        nxt = last_open + step
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.08)
    return rows


def rest_fill_missing(symbol: str, interval: str, out_root: Path, start_ms: int, end_ms: int, base_url: str) -> tuple[int, int]:
    merged = out_root / "klines" / symbol / interval / "merged.csv"
    existing = read_rows_csv(merged) if merged.exists() else []
    by_ts = {int(r[0]): r for r in existing}

    step = STEP_MS[interval]
    needed_ranges: list[tuple[int, int]] = []

    # Determine missing ranges from start..end inclusive
    ts = start_ms
    miss_start = None
    while ts <= end_ms:
        if ts not in by_ts:
            if miss_start is None:
                miss_start = ts
        else:
            if miss_start is not None:
                needed_ranges.append((miss_start, ts - step))
                miss_start = None
        ts += step
    if miss_start is not None:
        needed_ranges.append((miss_start, end_ms))

    fetched = 0
    for a, b in needed_ranges:
        try:
            batch = fetch_klines_rest_range(base_url, symbol, interval, a, b)
            for r in batch:
                by_ts[int(r[0])] = r
            fetched += len(batch)
            logging.info("[REST] %s %s filled %s..%s rows=%d", symbol, interval, a, b, len(batch))
        except Exception as exc:
            logging.error("[REST-ERR] %s %s range %s..%s: %s", symbol, interval, a, b, exc)

    final_rows = list(by_ts.values())
    cleaned, report = validate_rows(final_rows, interval)
    write_rows_csv(merged, cleaned)

    report_path = out_root / "klines" / "_reports" / f"{symbol}_{interval}_validation.json"
    write_validation_report(report_path, report)
    return fetched, report.gap_count


# ---------- symbol helpers ----------
def fetch_top_symbols(base_url: str, top_n: int) -> list[str]:
    payload = _rest_get(base_url, "/fapi/v1/ticker/24hr", {})
    items: list[tuple[str, float]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        s = str(row.get("symbol", ""))
        if not s.endswith("USDT") or "_" in s:
            continue
        try:
            qv = float(row.get("quoteVolume", 0.0))
        except Exception:
            continue
        items.append((s, qv))
    items.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in items[:top_n]]


# ---------- orchestrator ----------
def process_symbol_interval(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    out_root: Path,
    rest_fill: bool,
    base_url: str,
) -> None:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    for y, m in month_iter(start, end):
        try:
            download_month_zip(symbol, interval, y, m, out_root)
        except Exception as exc:
            logging.error("[MONTH-ERR] %s %s %04d-%02d %s", symbol, interval, y, m, exc)

    merged = merge_raw_files(symbol, interval, out_root, start_ms, end_ms)
    rows = read_rows_csv(merged)
    cleaned, report = validate_rows(rows, interval)
    write_rows_csv(merged, cleaned)

    report_path = out_root / "klines" / "_reports" / f"{symbol}_{interval}_validation.json"
    write_validation_report(report_path, report)

    if rest_fill:
        fetched, gaps = rest_fill_missing(symbol, interval, out_root, start_ms, end_ms, base_url)
        logging.info("[REST-SUMMARY] %s %s fetched=%d remaining_gaps=%d", symbol, interval, fetched, gaps)


def run_download(
    symbols: list[str],
    intervals: list[str],
    start: datetime,
    end: datetime,
    out_dir: str,
    max_workers: int,
    rest_fill: bool,
    base_url: str,
) -> None:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "symbols": symbols,
        "intervals": intervals,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    md_path = out_root / "klines" / "_reports" / "selected_symbols.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for s in symbols:
            for itv in intervals:
                futures.append(ex.submit(process_symbol_interval, s, itv, start, end, out_root, rest_fill, base_url))
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                logging.error("Task error: %s", exc)


def run_validate_or_merge_only(symbols: list[str], intervals: list[str], out_dir: str, mode: str) -> None:
    out_root = Path(out_dir)
    for s in symbols:
        for itv in intervals:
            merged = out_root / "klines" / s / itv / "merged.csv"
            if mode == "merge":
                # Merge all raw files available without date crop
                raw_dir = out_root / "klines" / s / itv / "raw"
                rows: list[list[str]] = []
                if raw_dir.exists():
                    for p in sorted(raw_dir.glob("*.csv")):
                        rows.extend(read_rows_csv(p))
                cleaned, rep = validate_rows(rows, itv)
                write_rows_csv(merged, cleaned)
                write_validation_report(out_root / "klines" / "_reports" / f"{s}_{itv}_validation.json", rep)
                logging.info("Merged %s %s rows=%d", s, itv, rep.total_rows)
            else:
                ok, rep = validate_csv_file(merged, itv)
                logging.info("Validate %s %s exists=%s rows=%d gaps=%d", s, itv, ok, rep.total_rows, rep.gap_count)
                write_validation_report(out_root / "klines" / "_reports" / f"{s}_{itv}_validation.json", rep)
