#!/usr/bin/env python3
"""
Live Binance USD-M Futures Trend-Pullback PAPER Simulator
=========================================================

- Continuous live market scanner (Top 5 USDT perpetual symbols by 24h quote volume).
- Strict global single-position mode (one open position at any moment).
- PAPER only (no real order placement).

How to run
----------
1) Default run:
   python futures_trend_pullback_telegram.py

2) Faster loop checks (for monitoring/testing):
   python futures_trend_pullback_telegram.py --loop-sleep-seconds 15

3) Allow only one closed trade then stop opening new trades:
   python futures_trend_pullback_telegram.py --max-total-trades 1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Any
from zoneinfo import ZoneInfo
from pathlib import Path

import requests
from requests import Response
from requests.exceptions import RequestException, Timeout

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# Decimal precision for financial calculations
getcontext().prec = 28

# -----------------------------
# Binance API constants
# -----------------------------
FUTURES_BASE = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
BOOK_TICKER_ENDPOINT = "/fapi/v1/ticker/bookTicker"
TICKER_24HR_ENDPOINT = "/fapi/v1/ticker/24hr"
PREMIUM_INDEX_ENDPOINT = "/fapi/v1/premiumIndex"

TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT = (5, 20)
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 20.0
JITTER_MAX = 0.5

CAIRO_TZ = ZoneInfo("Africa/Cairo")
DEC_ZERO = Decimal("0")
DEC_ONE = Decimal("1")
EPS = Decimal("1e-9")


# -----------------------------
# Data models
# -----------------------------
@dataclass
class Config:
    interval: str = "1m"
    candle_limit: int = 150
    top_n_symbols: int = 5

    ema_fast: int = 20
    ema_trend: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    volume_ma_period: int = 20

    use_rsi_filter: bool = True
    rsi_threshold: Decimal = Decimal("50")
    max_spread: Decimal = Decimal("0.1")

    atr_sl_mult: Decimal = Decimal("1")
    rr_ratio: Decimal = Decimal("1.5")

    initial_balance: Decimal = Decimal("100")
    margin_per_trade: Decimal = Decimal("90")
    leverage: Decimal = Decimal("3")
    fee_rate: Decimal = Decimal("0.0004")

    loop_sleep_seconds: int = 60
    summary_interval_seconds: int = 3600

    send_telegram: bool = True
    trades_csv: str = "backtest_results.csv"
    summary_csv: str = "hourly_summary.csv"

    max_loops: int | None = None
    max_total_trades: int | None = None


@dataclass
class Candle:
    open_time: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: int


@dataclass
class Position:
    symbol: str
    direction: str  # LONG | SHORT
    entry_time: int
    entry_fill_price: Decimal
    qty: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    entry_notional: Decimal
    entry_spread: Decimal
    volume_ratio: Decimal
    rsi_value: Decimal


@dataclass
class CandidateSignal:
    symbol: str
    direction: str
    score: Decimal
    spread: Decimal
    atr_value: Decimal
    volume_ratio: Decimal
    rsi_value: Decimal
    bid: Decimal
    ask: Decimal




@dataclass
class SignalCheck:
    candidate: CandidateSignal | None
    reject_reasons: list[str]

@dataclass
class Trade:
    timestamp: int
    symbol: str
    direction: str
    entry_fill_price: Decimal
    exit_fill_price: Decimal
    quantity: Decimal
    entry_notional: Decimal
    exit_notional: Decimal
    gross_pnl: Decimal
    fees: Decimal
    net_pnl: Decimal
    balance_after: Decimal
    exit_reason: str


@dataclass
class HourlySummary:
    timestamp: int
    profit_last_hour: Decimal
    loss_last_hour: Decimal
    net_last_hour: Decimal
    profit_total: Decimal
    loss_total: Decimal
    net_total: Decimal
    fees_last_hour: Decimal
    fees_total: Decimal
    total_trades: int
    wins: int
    losses: int
    win_rate: Decimal
    balance: Decimal


# -----------------------------
# Utility
# -----------------------------
def d(raw: Any) -> Decimal:
    return Decimal(str(raw))


def fmt2(val: Decimal) -> str:
    return str(val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def fmt4(val: Decimal) -> str:
    return str(val.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def fmt6(val: Decimal) -> str:
    return str(val.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def now_ms() -> int:
    return int(time.time() * 1000)


def cairo_now_text() -> str:
    return datetime.now(tz=CAIRO_TZ).strftime("%Y-%m-%d %H:%M:%S")


def ms_to_cairo_text(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(CAIRO_TZ).strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------
# Configuration and logging
# -----------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def maybe_load_dotenv() -> None:
    if load_dotenv is None:
        logging.info("python-dotenv not installed; skipping .env loading")
        return
    if load_dotenv():
        logging.info("Loaded environment variables from .env")


def parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_decimal(name: str, default: str) -> Decimal:
    raw = os.getenv(name)
    return d(raw) if raw and raw.strip() else d(default)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def load_config_from_env() -> Config:
    max_total_trades_raw = os.getenv("MAX_TOTAL_TRADES")
    max_total_trades = int(max_total_trades_raw) if max_total_trades_raw and max_total_trades_raw.strip() else None

    return Config(
        interval=os.getenv("INTERVAL", "1m"),
        candle_limit=env_int("CANDLE_LIMIT", 150),
        top_n_symbols=env_int("TOP_N_SYMBOLS", 5),
        ema_fast=env_int("EMA_FAST", 20),
        ema_trend=env_int("EMA_TREND", 50),
        rsi_period=env_int("RSI_PERIOD", 14),
        atr_period=env_int("ATR_PERIOD", 14),
        volume_ma_period=env_int("VOLUME_MA_PERIOD", 20),
        use_rsi_filter=parse_bool(os.getenv("USE_RSI_FILTER"), True),
        rsi_threshold=env_decimal("RSI_THRESHOLD", "50"),
        max_spread=env_decimal("MAX_SPREAD", "0.1"),
        atr_sl_mult=env_decimal("ATR_SL_MULT", "1"),
        rr_ratio=env_decimal("RR_RATIO", "1.5"),
        initial_balance=env_decimal("INITIAL_BALANCE", "100"),
        margin_per_trade=env_decimal("MARGIN_PER_TRADE", "90"),
        leverage=env_decimal("LEVERAGE", "3"),
        fee_rate=env_decimal("FEE_RATE", "0.0004"),
        loop_sleep_seconds=env_int("LOOP_SLEEP_SECONDS", 60),
        summary_interval_seconds=env_int("SUMMARY_INTERVAL_SECONDS", 3600),
        send_telegram=parse_bool(os.getenv("SEND_TELEGRAM"), True),
        trades_csv=os.getenv("TRADES_CSV", "backtest_results.csv"),
        summary_csv=os.getenv("SUMMARY_CSV", "hourly_summary.csv"),
        max_total_trades=max_total_trades,
    )


def validate_config(cfg: Config) -> None:
    if cfg.top_n_symbols <= 0:
        raise ValueError("top_n_symbols must be > 0")
    if cfg.candle_limit < 100:
        raise ValueError("candle_limit must be >= 100")
    if cfg.ema_fast <= 1 or cfg.ema_trend <= 1 or cfg.ema_fast >= cfg.ema_trend:
        raise ValueError("Invalid EMA settings")
    if cfg.rsi_period <= 1 or cfg.atr_period <= 1 or cfg.volume_ma_period <= 1:
        raise ValueError("Indicator periods must be > 1")
    if cfg.max_spread < DEC_ZERO:
        raise ValueError("max_spread must be >= 0")
    if cfg.atr_sl_mult <= DEC_ZERO or cfg.rr_ratio <= DEC_ZERO:
        raise ValueError("ATR multipliers must be > 0")
    if cfg.initial_balance <= DEC_ZERO:
        raise ValueError("initial_balance must be > 0")
    if cfg.margin_per_trade <= DEC_ZERO:
        raise ValueError("margin_per_trade must be > 0")
    if cfg.leverage <= DEC_ZERO:
        raise ValueError("leverage must be > 0")
    if cfg.fee_rate < DEC_ZERO:
        raise ValueError("fee_rate must be >= 0")
    if cfg.loop_sleep_seconds <= 0:
        raise ValueError("loop_sleep_seconds must be > 0")
    if cfg.summary_interval_seconds <= 0:
        raise ValueError("summary_interval_seconds must be > 0")
    if cfg.max_total_trades is not None and cfg.max_total_trades < 0:
        raise ValueError("max_total_trades must be >= 0")


# -----------------------------
# HTTP
# -----------------------------
def sleep_backoff(attempt: int) -> None:
    delay = min(INITIAL_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF) + random.uniform(0, JITTER_MAX)
    logging.info("Retrying in %.2f seconds", delay)
    time.sleep(delay)


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_payload: dict[str, Any] | None = None,
) -> Response:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.request(
                method=method,
                url=url,
                params=params,
                json=json_payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code in TRANSIENT_STATUSES and attempt < MAX_RETRIES:
                logging.warning("Transient HTTP %s for %s (attempt %d/%d)", response.status_code, url, attempt, MAX_RETRIES)
                sleep_backoff(attempt)
                continue
            return response
        except (Timeout, RequestException) as exc:
            last_exc = exc
            logging.warning("Request failed for %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                sleep_backoff(attempt)
                continue
    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url}") from last_exc


# -----------------------------
# Binance data
# -----------------------------
def fetch_top_symbols(session: requests.Session, top_n: int) -> list[str]:
    url = FUTURES_BASE + TICKER_24HR_ENDPOINT
    resp = request_with_retry(session, "GET", url)
    if resp.status_code != 200:
        raise RuntimeError(f"24hr ticker failed: HTTP {resp.status_code} - {resp.text}")

    payload = resp.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected /ticker/24hr payload")

    candidates: list[tuple[str, Decimal]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", ""))
        if not symbol.endswith("USDT") or "_" in symbol:
            continue
        try:
            qv = d(row.get("quoteVolume", "0"))
        except Exception:
            continue
        candidates.append((symbol, qv))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in candidates[:top_n]]


def fetch_klines(session: requests.Session, symbol: str, interval: str, limit: int) -> list[Candle]:
    url = FUTURES_BASE + KLINES_ENDPOINT
    resp = request_with_retry(session, "GET", url, params={"symbol": symbol, "interval": interval, "limit": limit})
    if resp.status_code != 200:
        raise RuntimeError(f"Klines failed for {symbol}: HTTP {resp.status_code} - {resp.text}")

    payload = resp.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Invalid klines payload for {symbol}")

    return [
        Candle(
            open_time=int(x[0]),
            open=d(x[1]),
            high=d(x[2]),
            low=d(x[3]),
            close=d(x[4]),
            volume=d(x[5]),
            close_time=int(x[6]),
        )
        for x in payload
    ]


def fetch_book_ticker(session: requests.Session, symbol: str) -> tuple[Decimal, Decimal, Decimal]:
    url = FUTURES_BASE + BOOK_TICKER_ENDPOINT
    resp = request_with_retry(session, "GET", url, params={"symbol": symbol})
    if resp.status_code != 200:
        raise RuntimeError(f"BookTicker failed for {symbol}: HTTP {resp.status_code} - {resp.text}")
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid bookTicker payload for {symbol}")

    bid = d(payload["bidPrice"])
    ask = d(payload["askPrice"])
    spread = ask - bid
    if spread < DEC_ZERO:
        raise RuntimeError(f"Negative spread for {symbol}")
    return bid, ask, spread


def fetch_mark_price(session: requests.Session, symbol: str) -> Decimal | None:
    url = FUTURES_BASE + PREMIUM_INDEX_ENDPOINT
    resp = request_with_retry(session, "GET", url, params={"symbol": symbol})
    if resp.status_code != 200:
        return None
    payload = resp.json()
    if not isinstance(payload, dict) or "markPrice" not in payload:
        return None
    return d(payload["markPrice"])


# -----------------------------
# Indicators (float math)
# -----------------------------
def ema(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1)
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    acc = sum(values[:period])
    out[period - 1] = acc / period
    for i in range(period, len(values)):
        acc += values[i] - values[i - period]
        out[i] = acc / period
    return out


def rsi(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return out

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    def calc(g: float, l: float) -> float:
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - (100.0 / (1 + rs))

    out[period] = calc(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        out[i] = calc(avg_gain, avg_loss)

    return out


def atr(candles: list[Candle], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(candles)
    if len(candles) <= period:
        return out

    tr_values: list[float] = []
    for i, candle in enumerate(candles):
        h, l = float(candle.high), float(candle.low)
        if i == 0:
            tr = h - l
        else:
            prev_close = float(candles[i - 1].close)
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        tr_values.append(tr)

    prev_atr = sum(tr_values[1 : period + 1]) / period
    out[period] = prev_atr
    for i in range(period + 1, len(candles)):
        prev_atr = ((prev_atr * (period - 1)) + tr_values[i]) / period
        out[i] = prev_atr

    return out


# -----------------------------
# Telegram & CSV
# -----------------------------
def telegram_credentials() -> tuple[str, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    return token, chat_id


def send_telegram_message(session: requests.Session, text: str) -> None:
    token, chat_id = telegram_credentials()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = request_with_retry(session, "POST", url, json_payload={"chat_id": chat_id, "text": text})
    if resp.status_code != 200:
        raise RuntimeError(f"Telegram sendMessage failed: HTTP {resp.status_code} - {resp.text}")


def init_trade_csv(path: str) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp_cairo",
                "symbol",
                "direction",
                "entry_fill_price",
                "exit_fill_price",
                "quantity",
                "entry_notional",
                "exit_notional",
                "gross_pnl",
                "fees",
                "net_pnl",
                "balance_after",
                "exit_reason",
            ]
        )


def append_trade_csv(path: str, trade: Trade) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                ms_to_cairo_text(trade.timestamp),
                trade.symbol,
                trade.direction,
                fmt6(trade.entry_fill_price),
                fmt6(trade.exit_fill_price),
                fmt6(trade.quantity),
                fmt6(trade.entry_notional),
                fmt6(trade.exit_notional),
                fmt6(trade.gross_pnl),
                fmt6(trade.fees),
                fmt6(trade.net_pnl),
                fmt6(trade.balance_after),
                trade.exit_reason,
            ]
        )


def init_summary_csv(path: str) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp_cairo",
                "profit_last_hour",
                "loss_last_hour",
                "net_last_hour",
                "profit_total",
                "loss_total",
                "net_total",
                "fees_last_hour",
                "fees_total",
                "total_trades",
                "wins",
                "losses",
                "win_rate_pct",
                "balance",
            ]
        )


def append_summary_csv(path: str, summary: HourlySummary) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                ms_to_cairo_text(summary.timestamp),
                fmt6(summary.profit_last_hour),
                fmt6(summary.loss_last_hour),
                fmt6(summary.net_last_hour),
                fmt6(summary.profit_total),
                fmt6(summary.loss_total),
                fmt6(summary.net_total),
                fmt6(summary.fees_last_hour),
                fmt6(summary.fees_total),
                summary.total_trades,
                summary.wins,
                summary.losses,
                fmt4(summary.win_rate),
                fmt6(summary.balance),
            ]
        )


# -----------------------------
# Strategy evaluation
# -----------------------------
def evaluate_candidate_signal(cfg: Config, symbol: str, candles: list[Candle], bid: Decimal, ask: Decimal, spread: Decimal) -> SignalCheck:
    closes = [float(c.close) for c in candles]
    volumes = [float(c.volume) for c in candles]

    ema_fast_vals = ema(closes, cfg.ema_fast)
    ema_trend_vals = ema(closes, cfg.ema_trend)
    rsi_vals = rsi(closes, cfg.rsi_period)
    atr_vals = atr(candles, cfg.atr_period)
    vma_vals = sma(volumes, cfg.volume_ma_period)

    i = len(candles) - 1
    c = candles[i]
    prev = candles[i - 1]

    e20 = ema_fast_vals[i]
    e50 = ema_trend_vals[i]
    r = rsi_vals[i]
    a = atr_vals[i]
    vma = vma_vals[i]
    prev_e20 = ema_fast_vals[i - 1]

    reasons: list[str] = []
    if any(v is None for v in [e20, e50, r, a, vma, prev_e20]):
        reasons.append("indicator_not_ready")
        return SignalCheck(candidate=None, reject_reasons=reasons)

    c_close = float(c.close)
    c_vol = float(c.volume)
    prev_close = float(prev.close)

    volume_ok = c_vol > float(vma)
    spread_ok = spread <= cfg.max_spread
    if not volume_ok:
        reasons.append(f"volume_filter_fail vol={c_vol:.4f} vma={float(vma):.4f}")
    if not spread_ok:
        reasons.append(f"spread_filter_fail spread={fmt4(spread)} max={fmt4(cfg.max_spread)}")
    if reasons:
        return SignalCheck(candidate=None, reject_reasons=reasons)

    long_trend = c_close > float(e50)
    short_trend = c_close < float(e50)
    long_trigger = prev_close <= float(prev_e20) and c_close > float(e20)
    short_trigger = prev_close >= float(prev_e20) and c_close < float(e20)

    rsi_val = d(r)
    long_rsi_ok = (rsi_val > cfg.rsi_threshold) if cfg.use_rsi_filter else True
    short_rsi_ok = (rsi_val < cfg.rsi_threshold) if cfg.use_rsi_filter else True

    direction: str | None = None
    if long_trend and long_trigger and long_rsi_ok:
        direction = "LONG"
    elif short_trend and short_trigger and short_rsi_ok:
        direction = "SHORT"

    if direction is None:
        if not long_trend and not short_trend:
            reasons.append("trend_filter_fail")
        if not long_trigger and not short_trigger:
            reasons.append("pullback_trigger_fail")
        if cfg.use_rsi_filter and not (long_rsi_ok or short_rsi_ok):
            reasons.append(f"rsi_filter_fail rsi={fmt4(rsi_val)} threshold={fmt4(cfg.rsi_threshold)}")
        return SignalCheck(candidate=None, reject_reasons=reasons or ["no_direction"])

    atr_val = d(a)
    if atr_val <= DEC_ZERO:
        return SignalCheck(candidate=None, reject_reasons=["atr_non_positive"])

    volume_ratio = d(c_vol) / d(vma)
    atr_pct = atr_val / c.close if c.close > DEC_ZERO else DEC_ZERO
    score = (volume_ratio / (spread + EPS)) * (DEC_ONE + atr_pct)

    return SignalCheck(
        candidate=CandidateSignal(
            symbol=symbol,
            direction=direction,
            score=score,
            spread=spread,
            atr_value=atr_val,
            volume_ratio=volume_ratio,
            rsi_value=rsi_val,
            bid=bid,
            ask=ask,
        ),
        reject_reasons=[],
    )

def create_position_from_candidate(cfg: Config, candidate: CandidateSignal, latest_close_time: int) -> Position | None:
    entry_fill = candidate.ask if candidate.direction == "LONG" else candidate.bid
    if entry_fill <= DEC_ZERO:
        return None

    notional = cfg.margin_per_trade * cfg.leverage  # must remain 90*3 by default
    qty = notional / entry_fill
    if qty <= DEC_ZERO:
        return None

    risk = cfg.atr_sl_mult * candidate.atr_value
    if risk <= DEC_ZERO:
        return None

    if candidate.direction == "LONG":
        stop_loss = entry_fill - risk
        take_profit = entry_fill + (cfg.rr_ratio * risk)
    else:
        stop_loss = entry_fill + risk
        take_profit = entry_fill - (cfg.rr_ratio * risk)

    return Position(
        symbol=candidate.symbol,
        direction=candidate.direction,
        entry_time=latest_close_time,
        entry_fill_price=entry_fill,
        qty=qty,
        stop_loss=stop_loss,
        take_profit=take_profit,
        entry_notional=entry_fill * qty,
        entry_spread=candidate.spread,
        volume_ratio=candidate.volume_ratio,
        rsi_value=candidate.rsi_value,
    )


def evaluate_live_exit(position: Position, bid: Decimal, ask: Decimal, fee_rate: Decimal, balance: Decimal) -> tuple[Trade | None, Decimal]:
    exit_reason = ""
    exit_fill: Decimal | None = None

    # Conservative trigger + executable fill
    if position.direction == "LONG":
        if bid <= position.stop_loss:
            exit_reason = "SL"
            exit_fill = bid
        elif bid >= position.take_profit:
            exit_reason = "TP"
            exit_fill = bid
    else:
        if ask >= position.stop_loss:
            exit_reason = "SL"
            exit_fill = ask
        elif ask <= position.take_profit:
            exit_reason = "TP"
            exit_fill = ask

    if exit_fill is None:
        return None, balance

    exit_notional = exit_fill * position.qty
    fees = (position.entry_notional * fee_rate) + (exit_notional * fee_rate)

    if position.direction == "LONG":
        gross = (exit_fill - position.entry_fill_price) * position.qty
        dir_ar = "شراء"
    else:
        gross = (position.entry_fill_price - exit_fill) * position.qty
        dir_ar = "بيع"

    net = gross - fees
    new_balance = balance + net

    trade = Trade(
        timestamp=now_ms(),
        symbol=position.symbol,
        direction=dir_ar,
        entry_fill_price=position.entry_fill_price,
        exit_fill_price=exit_fill,
        quantity=position.qty,
        entry_notional=position.entry_notional,
        exit_notional=exit_notional,
        gross_pnl=gross,
        fees=fees,
        net_pnl=net,
        balance_after=new_balance,
        exit_reason=exit_reason,
    )
    return trade, new_balance


def estimate_unrealized(position: Position, bid: Decimal, ask: Decimal, fee_rate: Decimal) -> tuple[Decimal, Decimal]:
    if position.direction == "LONG":
        exit_fill = bid
        gross = (exit_fill - position.entry_fill_price) * position.qty
    else:
        exit_fill = ask
        gross = (position.entry_fill_price - exit_fill) * position.qty

    exit_notional = exit_fill * position.qty
    est_fees = (position.entry_notional * fee_rate) + (exit_notional * fee_rate)
    net_est = gross - est_fees
    return gross, net_est


# -----------------------------
# Telegram message builders (Arabic)
# -----------------------------
def send_entry_alert_ar(session: requests.Session, cfg: Config, pos: Position, candidate: CandidateSignal) -> None:
    msg = (
        "🟢 فتح صفقة (تجريبي)\n"
        f"🕒 الوقت (القاهرة): {cairo_now_text()}\n"
        f"📌 الرمز: {pos.symbol}\n"
        f"📈 الاتجاه: {'شراء' if pos.direction == 'LONG' else 'بيع'}\n"
        f"⚙️ الرافعة: x{fmt2(cfg.leverage)} | الهامش: {fmt2(cfg.margin_per_trade)} USDT | النوتيشنال: {fmt2(cfg.margin_per_trade * cfg.leverage)} USDT\n"
        f"💲 سعر الدخول (تنفيذ): {fmt4(pos.entry_fill_price)} | Bid: {fmt4(candidate.bid)} | Ask: {fmt4(candidate.ask)}\n"
        f"↔️ السبريد: {fmt4(candidate.spread)}\n"
        f"🛑 وقف الخسارة: {fmt4(pos.stop_loss)} | 🎯 جني الأرباح: {fmt4(pos.take_profit)}\n"
        f"📊 Volume/VMA20: {fmt4(pos.volume_ratio)} | RSI: {fmt4(pos.rsi_value)} {'(مفعل)' if cfg.use_rsi_filter else '(غير مفعل)'}"
    )
    send_telegram_message(session, msg)


def send_exit_alert_ar(session: requests.Session, trade: Trade) -> None:
    msg = (
        "🔴 إغلاق صفقة\n"
        f"🕒 الوقت (القاهرة): {cairo_now_text()}\n"
        f"📌 الرمز: {trade.symbol} | الاتجاه: {trade.direction}\n"
        f"💲 دخول: {fmt4(trade.entry_fill_price)} | خروج: {fmt4(trade.exit_fill_price)}\n"
        f"📦 الكمية: {fmt6(trade.quantity)}\n"
        f"📈 الربح/الخسارة الإجمالي: {fmt4(trade.gross_pnl)} USDT\n"
        f"💸 العمولات: {fmt4(trade.fees)} USDT\n"
        f"✅ الصافي: {fmt4(trade.net_pnl)} USDT\n"
        f"💼 الرصيد بعد الإغلاق: {fmt4(trade.balance_after)} USDT\n"
        f"🧾 سبب الإغلاق: {trade.exit_reason}"
    )
    send_telegram_message(session, msg)


def send_hourly_summary_ar(
    session: requests.Session,
    summary: HourlySummary,
    open_position: Position | None,
    unrealized_net: Decimal | None,
) -> None:
    if open_position is None:
        pos_line = "🟢 حالة الصفقة الحالية: لا توجد صفقة مفتوحة"
    else:
        pos_line = (
            "🟠 حالة الصفقة الحالية: "
            f"{open_position.symbol} | {'شراء' if open_position.direction == 'LONG' else 'بيع'} "
            f"| PnL غير محقق: {fmt4(unrealized_net or DEC_ZERO)} USDT"
        )

    msg = (
        "📊 الملخص الساعي\n"
        f"🕒 الوقت (القاهرة): {cairo_now_text()}\n"
        f"ربح آخر ساعة: {fmt4(summary.profit_last_hour)} USDT\n"
        f"خسارة آخر ساعة: {fmt4(summary.loss_last_hour)} USDT\n"
        f"صافي آخر ساعة: {fmt4(summary.net_last_hour)} USDT\n"
        f"إجمالي الربح منذ البداية: {fmt4(summary.profit_total)} USDT\n"
        f"إجمالي الخسارة منذ البداية: {fmt4(summary.loss_total)} USDT\n"
        f"صافي الإجمالي منذ البداية: {fmt4(summary.net_total)} USDT\n"
        f"العمولات (آخر ساعة): {fmt4(summary.fees_last_hour)} USDT\n"
        f"العمولات (إجمالي): {fmt4(summary.fees_total)} USDT\n"
        f"عدد الصفقات: {summary.total_trades} | رابحة: {summary.wins} | خاسرة: {summary.losses} | نسبة النجاح: {fmt2(summary.win_rate)}%\n"
        f"💼 الرصيد الحالي: {fmt4(summary.balance)} USDT\n"
        f"{pos_line}"
    )
    send_telegram_message(session, msg)


# -----------------------------
# Summary / heartbeat
# -----------------------------
def build_hourly_summary(trades: list[Trade], balance: Decimal, ts_ms: int) -> HourlySummary:
    one_hour_ago = ts_ms - 3_600_000

    last_hour = [t for t in trades if t.timestamp >= one_hour_ago]

    profit_last_hour = sum((t.net_pnl for t in last_hour if t.net_pnl > DEC_ZERO), DEC_ZERO)
    loss_last_hour = sum((t.net_pnl for t in last_hour if t.net_pnl < DEC_ZERO), DEC_ZERO)
    net_last_hour = profit_last_hour + loss_last_hour

    profit_total = sum((t.net_pnl for t in trades if t.net_pnl > DEC_ZERO), DEC_ZERO)
    loss_total = sum((t.net_pnl for t in trades if t.net_pnl < DEC_ZERO), DEC_ZERO)
    net_total = profit_total + loss_total

    fees_last_hour = sum((t.fees for t in last_hour), DEC_ZERO)
    fees_total = sum((t.fees for t in trades), DEC_ZERO)

    wins = sum(1 for t in trades if t.net_pnl > DEC_ZERO)
    losses = sum(1 for t in trades if t.net_pnl < DEC_ZERO)
    total = len(trades)
    win_rate = (d(wins) / d(total) * d("100")) if total > 0 else DEC_ZERO

    return HourlySummary(
        timestamp=ts_ms,
        profit_last_hour=profit_last_hour,
        loss_last_hour=loss_last_hour,
        net_last_hour=net_last_hour,
        profit_total=profit_total,
        loss_total=loss_total,
        net_total=net_total,
        fees_last_hour=fees_last_hour,
        fees_total=fees_total,
        total_trades=total,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        balance=balance,
    )


def heartbeat_log(
    cfg: Config,
    open_position: Position | None,
    mark_price: Decimal | None,
    bid: Decimal | None,
    ask: Decimal | None,
    unrealized_net: Decimal | None,
    sleep_seconds: float,
) -> None:
    now_local = datetime.now(tz=CAIRO_TZ)
    next_check = now_local.timestamp() + sleep_seconds
    next_check_text = datetime.fromtimestamp(next_check, tz=CAIRO_TZ).strftime("%Y-%m-%d %H:%M:%S")

    if open_position is None:
        pos_text = "لا توجد صفقة مفتوحة"
    else:
        mid = ((bid + ask) / d("2")) if bid is not None and ask is not None else None
        current = mark_price if mark_price is not None else mid
        pos_text = (
            f"صفقة مفتوحة: {open_position.symbol} | {'شراء' if open_position.direction == 'LONG' else 'بيع'} "
            f"| دخول={fmt4(open_position.entry_fill_price)} "
            f"| السعر الحالي={fmt4(current) if current is not None else 'N/A'} "
            f"| PnL غير محقق={fmt4(unrealized_net or DEC_ZERO)} USDT"
        )

    logging.info(
        "Heartbeat | القاهرة=%s | %s | النوم %.1f ثانية | الفحص القادم %s",
        now_local.strftime("%Y-%m-%d %H:%M:%S"),
        pos_text,
        sleep_seconds,
        next_check_text,
    )


# -----------------------------
# Main live loop
# -----------------------------
def run_live_paper(session: requests.Session, cfg: Config) -> None:
    balance = cfg.initial_balance
    trades: list[Trade] = []
    open_position: Position | None = None

    init_trade_csv(cfg.trades_csv)
    init_summary_csv(cfg.summary_csv)

    # Align summary timer to wall-clock interval boundary
    now_epoch = int(time.time())
    next_summary_epoch = ((now_epoch // cfg.summary_interval_seconds) + 1) * cfg.summary_interval_seconds

    loop_count = 0

    logging.info(
        "Starting PAPER live loop | balance=%s | margin=%s | leverage=%s | notional=%s | fee_rate=%s",
        fmt2(cfg.initial_balance),
        fmt2(cfg.margin_per_trade),
        fmt2(cfg.leverage),
        fmt2(cfg.margin_per_trade * cfg.leverage),
        str(cfg.fee_rate),
    )

    while True:
        loop_count += 1
        loop_start = time.time()

        hb_bid: Decimal | None = None
        hb_ask: Decimal | None = None
        hb_mark: Decimal | None = None
        hb_unrealized_net: Decimal | None = None

        try:
            top_symbols = fetch_top_symbols(session, cfg.top_n_symbols)
            logging.info("Top %d symbols by 24h quote volume: %s", cfg.top_n_symbols, ", ".join(top_symbols))

            # 1) Manage open position first (single global position enforcement)
            if open_position is not None:
                bid, ask, _ = fetch_book_ticker(session, open_position.symbol)
                hb_bid, hb_ask = bid, ask
                hb_mark = fetch_mark_price(session, open_position.symbol)

                trade, balance = evaluate_live_exit(open_position, bid, ask, cfg.fee_rate, balance)

                # heartbeat unrealized when still open
                if trade is None:
                    _, hb_unrealized_net = estimate_unrealized(open_position, bid, ask, cfg.fee_rate)
                else:
                    trades.append(trade)
                    append_trade_csv(cfg.trades_csv, trade)
                    logging.info(
                        "Closed %s %s | entry=%s exit=%s gross=%s fees=%s net=%s balance=%s reason=%s",
                        trade.symbol,
                        trade.direction,
                        fmt4(trade.entry_fill_price),
                        fmt4(trade.exit_fill_price),
                        fmt4(trade.gross_pnl),
                        fmt4(trade.fees),
                        fmt4(trade.net_pnl),
                        fmt4(trade.balance_after),
                        trade.exit_reason,
                    )
                    if cfg.send_telegram:
                        try:
                            send_exit_alert_ar(session, trade)
                        except Exception as exc:  # noqa: BLE001
                            logging.error("Failed to send Telegram exit alert: %s", exc)
                    open_position = None

            # 2) If no position open, evaluate ALL top symbols and choose best score
            can_open_more_by_count = cfg.max_total_trades is None or len(trades) < cfg.max_total_trades
            if open_position is None and not can_open_more_by_count:
                logging.info("Trade cap reached (max_total_trades=%s). New entries are disabled.", cfg.max_total_trades)

            if open_position is None and can_open_more_by_count:
                if balance < cfg.margin_per_trade:
                    logging.info(
                        "Skipping entries: balance (%s) < margin_per_trade (%s)",
                        fmt2(balance),
                        fmt2(cfg.margin_per_trade),
                    )
                else:
                    candidates: list[tuple[CandidateSignal, int]] = []
                    for symbol in top_symbols:
                        try:
                            candles = fetch_klines(session, symbol, cfg.interval, cfg.candle_limit)
                            bid, ask, spread = fetch_book_ticker(session, symbol)
                            check = evaluate_candidate_signal(cfg, symbol, candles, bid, ask, spread)
                            candidate = check.candidate
                            if candidate is not None:
                                candidates.append((candidate, candles[-1].close_time))
                                logging.info(
                                    "Candidate %s | dir=%s score=%s vol_ratio=%s spread=%s atr=%s rsi=%s",
                                    candidate.symbol,
                                    candidate.direction,
                                    fmt4(candidate.score),
                                    fmt4(candidate.volume_ratio),
                                    fmt4(candidate.spread),
                                    fmt4(candidate.atr_value),
                                    fmt4(candidate.rsi_value),
                                )
                            else:
                                logging.info("Rejected %s signal: %s", symbol, "; ".join(check.reject_reasons))
                        except Exception as exc:  # noqa: BLE001
                            logging.warning("Signal scan failed for %s: %s", symbol, exc)

                    if candidates:
                        chosen, close_time = max(candidates, key=lambda x: x[0].score)
                        position = create_position_from_candidate(cfg, chosen, close_time)
                        if position is not None:
                            if open_position is not None:
                                logging.error("Single-position guard violated; refusing new entry on %s", chosen.symbol)
                            else:
                                open_position = position
                            logging.info(
                                "Chosen %s | dir=%s score=%s (vol_ratio=%s spread=%s atr=%s) | entry_fill=%s",
                                chosen.symbol,
                                chosen.direction,
                                fmt4(chosen.score),
                                fmt4(chosen.volume_ratio),
                                fmt4(chosen.spread),
                                fmt4(chosen.atr_value),
                                fmt4(position.entry_fill_price),
                            )
                            if cfg.send_telegram:
                                try:
                                    send_entry_alert_ar(session, cfg, position, chosen)
                                except Exception as exc:  # noqa: BLE001
                                    logging.error("Failed to send Telegram entry alert: %s", exc)
                    else:
                        logging.info("No valid entry candidates this loop.")

            # 3) Hourly summary (exactly at interval boundary or first pass after it)
            now_epoch = int(time.time())
            if now_epoch >= next_summary_epoch:
                summary = build_hourly_summary(trades, balance, now_ms())
                append_summary_csv(cfg.summary_csv, summary)

                if open_position is not None:
                    try:
                        bid, ask, _ = fetch_book_ticker(session, open_position.symbol)
                        _, hb_unrealized_net = estimate_unrealized(open_position, bid, ask, cfg.fee_rate)
                        hb_bid, hb_ask = bid, ask
                        hb_mark = fetch_mark_price(session, open_position.symbol)
                    except Exception as exc:  # noqa: BLE001
                        logging.warning("Could not refresh open-position metrics for summary: %s", exc)

                logging.info(
                    "Hourly summary | net_last_hour=%s net_total=%s fees_hour=%s fees_total=%s trades=%d win_rate=%s%% balance=%s",
                    fmt4(summary.net_last_hour),
                    fmt4(summary.net_total),
                    fmt4(summary.fees_last_hour),
                    fmt4(summary.fees_total),
                    summary.total_trades,
                    fmt2(summary.win_rate),
                    fmt4(summary.balance),
                )

                if cfg.send_telegram:
                    try:
                        send_hourly_summary_ar(session, summary, open_position, hb_unrealized_net)
                    except Exception as exc:  # noqa: BLE001
                        logging.error("Failed to send Telegram hourly summary: %s", exc)

                while next_summary_epoch <= now_epoch:
                    next_summary_epoch += cfg.summary_interval_seconds

        except Exception as exc:  # noqa: BLE001
            logging.error("Main loop error: %s", exc)

        elapsed = time.time() - loop_start
        sleep_seconds = max(1.0, cfg.loop_sleep_seconds - elapsed)

        heartbeat_log(
            cfg,
            open_position,
            hb_mark,
            hb_bid,
            hb_ask,
            hb_unrealized_net,
            sleep_seconds,
        )

        if cfg.max_loops is not None and loop_count >= cfg.max_loops:
            logging.info("Reached max_loops=%d, exiting.", cfg.max_loops)
            break

        time.sleep(sleep_seconds)




# -----------------------------
# Backtest data loaders
# -----------------------------
def load_local_klines(data_dir: str, symbol: str, interval: str) -> list[Candle]:
    base = Path(data_dir) / "klines" / symbol / interval
    parquet_path = base / "merged.parquet"
    csv_path = base / "merged.csv"

    rows: list[list[str]] = []
    if parquet_path.exists():
        try:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(parquet_path)
            rows = df.astype(str).values.tolist()
        except Exception:
            rows = []

    if not rows:
        if not csv_path.exists():
            raise FileNotFoundError(f"Local merged data not found: {csv_path}")
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 12:
                    rows.append(row[:12])

    candles: list[Candle] = []
    for r in rows:
        candles.append(
            Candle(
                open_time=int(r[0]),
                open=d(r[1]),
                high=d(r[2]),
                low=d(r[3]),
                close=d(r[4]),
                volume=d(r[5]),
                close_time=int(r[6]),
            )
        )
    return candles


def fetch_klines_rest_for_backtest(session: requests.Session, symbol: str, interval: str, limit: int = 1500) -> list[Candle]:
    resp = request_with_retry(
        session,
        "GET",
        f"{FUTURES_BASE}{KLINES_ENDPOINT}",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Backtest klines fetch failed: HTTP {resp.status_code} - {resp.text}")
    payload = resp.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected klines payload")
    return [
        Candle(
            open_time=int(x[0]),
            open=d(x[1]),
            high=d(x[2]),
            low=d(x[3]),
            close=d(x[4]),
            volume=d(x[5]),
            close_time=int(x[6]),
        )
        for x in payload
    ]


def run_backtest_mode(session: requests.Session, cfg: Config, symbol: str, data_dir: str | None) -> None:
    candles = load_local_klines(data_dir, symbol, cfg.interval) if data_dir else fetch_klines_rest_for_backtest(session, symbol, cfg.interval)
    if len(candles) < 100:
        raise RuntimeError("Not enough candles for backtest")

    closes = [float(c.close) for c in candles]
    volumes = [float(c.volume) for c in candles]
    e20 = ema(closes, cfg.ema_fast)
    e50 = ema(closes, cfg.ema_trend)
    rsi_vals = rsi(closes, cfg.rsi_period)
    atr_vals = atr(candles, cfg.atr_period)
    vma = sma(volumes, cfg.volume_ma_period)

    balance = cfg.initial_balance
    pos: Position | None = None
    trades: list[Trade] = []

    for i in range(max(cfg.ema_trend, cfg.atr_period, cfg.volume_ma_period, cfg.rsi_period) + 1, len(candles)):
        c = candles[i]
        prev = candles[i - 1]

        if pos is not None:
            # candle-based conservative exit in backtest mode
            if pos.direction == "LONG":
                bid = c.low
                ask = c.high
            else:
                bid = c.low
                ask = c.high
            trade, balance = evaluate_live_exit(pos, bid, ask, cfg.fee_rate, balance)
            if trade is not None:
                trades.append(trade)
                pos = None

        if pos is not None:
            continue

        if cfg.max_total_trades is not None and len(trades) >= cfg.max_total_trades:
            continue

        if balance < cfg.margin_per_trade:
            continue

        if any(x is None for x in [e20[i], e50[i], rsi_vals[i], atr_vals[i], vma[i], e20[i - 1]]):
            continue

        spread = DEC_ZERO
        volume_ok = d(volumes[i]) > d(vma[i])
        spread_ok = spread <= cfg.max_spread
        if not volume_ok or not spread_ok:
            continue

        long_trend = closes[i] > float(e50[i])
        short_trend = closes[i] < float(e50[i])
        long_trigger = float(prev.close) <= float(e20[i - 1]) and closes[i] > float(e20[i])
        short_trigger = float(prev.close) >= float(e20[i - 1]) and closes[i] < float(e20[i])

        r = d(rsi_vals[i])
        long_rsi_ok = (r > cfg.rsi_threshold) if cfg.use_rsi_filter else True
        short_rsi_ok = (r < cfg.rsi_threshold) if cfg.use_rsi_filter else True

        direction = None
        if long_trend and long_trigger and long_rsi_ok:
            direction = "LONG"
        elif short_trend and short_trigger and short_rsi_ok:
            direction = "SHORT"
        if direction is None:
            continue

        atr_val = d(atr_vals[i])
        cand = CandidateSignal(
            symbol=symbol,
            direction=direction,
            score=d("1"),
            spread=spread,
            atr_value=atr_val,
            volume_ratio=d(volumes[i]) / d(vma[i]),
            rsi_value=r,
            bid=c.close,
            ask=c.close,
        )
        pos = create_position_from_candidate(cfg, cand, c.close_time)

    total_net = sum((t.net_pnl for t in trades), DEC_ZERO)
    wins = sum(1 for t in trades if t.net_pnl > DEC_ZERO)
    print("=== Backtest Summary ===")
    print(f"Symbol: {symbol}")
    print(f"Interval: {cfg.interval}")
    print(f"Trades: {len(trades)}")
    print(f"Win rate: {(wins/len(trades)*100 if trades else 0):.2f}%")
    print(f"Net PnL: {fmt4(total_net)} USDT")
    print(f"Final balance: {fmt4(balance)} USDT")


def write_runtime_system_report() -> None:
    report = {
        "single_position_enforced": True,
        "defaults": {
            "initial_balance": "100",
            "margin_per_trade": "90",
            "leverage": "3",
            "fee_rate": "0.0004",
            "top_n_symbols": 5,
            "summary_interval_seconds": 3600,
        },
        "examples": {
            "download": "python download_um_futures_klines.py --symbols BTCUSDT,ETHUSDT --intervals 15m,1h --start 2023-02-17 --end 2026-02-17 --out data/um_futures",
            "backtest": "python futures_trend_pullback_telegram.py --mode backtest --symbol BTCUSDT --interval 1h --data-dir data/um_futures",
            "live": "python futures_trend_pullback_telegram.py --mode live",
        },
    }
    out = Path("runtime_system_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Runtime system report saved: %s", out)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live/Backtest Binance USD-M futures PAPER simulator with Arabic Telegram alerts")
    p.add_argument("--mode", type=str, default="live", choices=["live", "backtest"])
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--data-dir", type=str, help="Path like data/um_futures for local merged klines")
    p.add_argument("--interval", type=str)
    p.add_argument("--candle-limit", type=int)
    p.add_argument("--top-n-symbols", type=int)
    p.add_argument("--fee-rate", type=str)
    p.add_argument("--leverage", type=str)
    p.add_argument("--margin-per-trade", type=str)
    p.add_argument("--max-spread", type=str)
    p.add_argument("--summary-interval-seconds", type=int)
    p.add_argument("--loop-sleep-seconds", type=int)
    p.add_argument("--max-total-trades", type=int)
    p.add_argument("--no-telegram", action="store_true")
    p.add_argument("--max-loops", type=int, help="testing helper")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = load_config_from_env()

    if args.interval:
        cfg.interval = args.interval
    if args.candle_limit is not None:
        cfg.candle_limit = args.candle_limit
    if args.top_n_symbols is not None:
        cfg.top_n_symbols = args.top_n_symbols
    if args.fee_rate is not None:
        cfg.fee_rate = d(args.fee_rate)
    if args.leverage is not None:
        cfg.leverage = d(args.leverage)
    if args.margin_per_trade is not None:
        cfg.margin_per_trade = d(args.margin_per_trade)
    if args.max_spread is not None:
        cfg.max_spread = d(args.max_spread)
    if args.summary_interval_seconds is not None:
        cfg.summary_interval_seconds = args.summary_interval_seconds
    if args.loop_sleep_seconds is not None:
        cfg.loop_sleep_seconds = args.loop_sleep_seconds
    if args.max_total_trades is not None:
        cfg.max_total_trades = args.max_total_trades
    if args.max_loops is not None:
        cfg.max_loops = args.max_loops
    if args.no_telegram:
        cfg.send_telegram = False

    validate_config(cfg)
    return cfg


def main() -> int:
    setup_logging()
    maybe_load_dotenv()

    try:
        args = parse_args()
        cfg = build_config(args)
        write_runtime_system_report()
        with requests.Session() as session:
            if args.mode == "backtest":
                run_backtest_mode(session, cfg, args.symbol.upper(), args.data_dir)
            else:
                run_live_paper(session, cfg)
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
