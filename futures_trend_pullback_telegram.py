#!/usr/bin/env python3
"""
Live Binance Futures Trend-Pullback Trader + Arabic Telegram Alerts
===================================================================

Features
--------
- Runs continuously on live Binance USDⓈ-M Futures data.
- Selects top 5 USDT perpetual symbols by 24h quote volume.
- Evaluates original trend-pullback logic (EMA20/EMA50 + volume + spread + optional RSI + ATR exits).
- Keeps at most one open position across all symbols.
- Uses 3x leverage sizing by default: 90 USD margin * 3 = 270 USD notional.
- Tracks commission (default 0.04% per side), net PnL, and running account balance.
- Sends Arabic Telegram alerts on entry, exit, and periodic summaries.
- Writes detailed trades CSV and hourly summary CSV.

Important
---------
This script is for strategy simulation / paper execution flow against live market data.
It does NOT place exchange orders.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
from requests import Response
from requests.exceptions import RequestException, Timeout

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# -----------------------------
# Binance API constants
# -----------------------------
FUTURES_BASE = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
BOOK_TICKER_ENDPOINT = "/fapi/v1/ticker/bookTicker"
TICKER_24HR_ENDPOINT = "/fapi/v1/ticker/24hr"

TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT = (5, 20)  # connect, read
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 20.0
JITTER_MAX = 0.5


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
    rsi_threshold: float = 50.0
    max_spread: float = 0.1

    atr_sl_mult: float = 1.0
    rr_ratio: float = 1.5

    initial_balance: float = 100.0
    margin_per_trade: float = 90.0
    leverage: float = 3.0
    fee_rate: float = 0.0004  # 0.04% per side

    loop_sleep_seconds: int = 60
    summary_interval_seconds: int = 3600

    send_telegram: bool = True
    trades_csv: str = "backtest_results.csv"
    summary_csv: str = "hourly_summary.csv"

    # optional test helper to stop loop; None = infinite
    max_loops: int | None = None


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


@dataclass
class Position:
    symbol: str
    direction: str  # LONG | SHORT
    entry_time: int
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float


@dataclass
class Trade:
    timestamp: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    balance: float
    exit_reason: str


@dataclass
class HourlySummary:
    timestamp: int
    pnl_total: float
    pnl_last_hour: float
    total_trades: int
    wins: int
    losses: int
    balance: float


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
        logging.info("python-dotenv is not installed; skipping .env loading")
        return
    if load_dotenv():
        logging.info("Loaded environment variables from .env")


def parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw and raw.strip() else default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def load_config_from_env() -> Config:
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
        rsi_threshold=env_float("RSI_THRESHOLD", 50.0),
        max_spread=env_float("MAX_SPREAD", 0.1),
        atr_sl_mult=env_float("ATR_SL_MULT", 1.0),
        rr_ratio=env_float("RR_RATIO", 1.5),
        initial_balance=env_float("INITIAL_BALANCE", 100.0),
        margin_per_trade=env_float("MARGIN_PER_TRADE", 90.0),
        leverage=env_float("LEVERAGE", 3.0),
        fee_rate=env_float("FEE_RATE", 0.0004),
        loop_sleep_seconds=env_int("LOOP_SLEEP_SECONDS", 60),
        summary_interval_seconds=env_int("SUMMARY_INTERVAL_SECONDS", 3600),
        send_telegram=parse_bool(os.getenv("SEND_TELEGRAM"), True),
        trades_csv=os.getenv("TRADES_CSV", "backtest_results.csv"),
        summary_csv=os.getenv("SUMMARY_CSV", "hourly_summary.csv"),
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
    if cfg.max_spread < 0:
        raise ValueError("max_spread must be >= 0")
    if cfg.atr_sl_mult <= 0 or cfg.rr_ratio <= 0:
        raise ValueError("ATR multipliers must be > 0")
    if cfg.initial_balance <= 0:
        raise ValueError("initial_balance must be > 0")
    if cfg.margin_per_trade <= 0:
        raise ValueError("margin_per_trade must be > 0")
    if cfg.leverage <= 0:
        raise ValueError("leverage must be > 0")
    if cfg.fee_rate < 0:
        raise ValueError("fee_rate must be >= 0")
    if cfg.loop_sleep_seconds <= 0:
        raise ValueError("loop_sleep_seconds must be > 0")
    if cfg.summary_interval_seconds <= 0:
        raise ValueError("summary_interval_seconds must be > 0")


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
            if response.status_code in TRANSIENT_STATUSES:
                logging.warning(
                    "Transient HTTP %s for %s (attempt %d/%d)",
                    response.status_code,
                    url,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt < MAX_RETRIES:
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
        raise RuntimeError(f"24hr ticker request failed: HTTP {resp.status_code} - {resp.text}")

    payload = resp.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected 24hr ticker payload format")

    candidates: list[tuple[str, float]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", ""))
        if not symbol.endswith("USDT"):
            continue
        # Exclude dated contracts like BTCUSDT_240329
        if "_" in symbol:
            continue
        try:
            quote_volume = float(row.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            continue
        candidates.append((symbol, quote_volume))

    if not candidates:
        raise RuntimeError("No USDT perpetual futures symbols found in 24hr ticker")

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [symbol for symbol, _ in candidates[:top_n]]


def fetch_klines(session: requests.Session, symbol: str, interval: str, limit: int) -> list[Candle]:
    url = FUTURES_BASE + KLINES_ENDPOINT
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = request_with_retry(session, "GET", url, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"Klines request failed for {symbol}: HTTP {resp.status_code} - {resp.text}")

    payload = resp.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Invalid/empty kline payload for {symbol}")

    candles = [
        Candle(
            open_time=int(x[0]),
            open=float(x[1]),
            high=float(x[2]),
            low=float(x[3]),
            close=float(x[4]),
            volume=float(x[5]),
            close_time=int(x[6]),
        )
        for x in payload
    ]
    return candles


def fetch_spread(session: requests.Session, symbol: str) -> float:
    url = FUTURES_BASE + BOOK_TICKER_ENDPOINT
    resp = request_with_retry(session, "GET", url, params={"symbol": symbol})
    if resp.status_code != 200:
        raise RuntimeError(f"BookTicker request failed for {symbol}: HTTP {resp.status_code} - {resp.text}")

    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid bookTicker payload for {symbol}")

    bid = float(payload["bidPrice"])
    ask = float(payload["askPrice"])
    spread = ask - bid
    if spread < 0:
        raise RuntimeError(f"Negative spread for {symbol}: {spread}")
    return spread


# -----------------------------
# Indicators
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

    def calc_rsi(g: float, l: float) -> float:
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - (100.0 / (1.0 + rs))

    out[period] = calc_rsi(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        out[i] = calc_rsi(avg_gain, avg_loss)

    return out


def atr(candles: list[Candle], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(candles)
    if len(candles) <= period:
        return out

    tr_values: list[float] = []
    for i, candle in enumerate(candles):
        if i == 0:
            tr = candle.high - candle.low
        else:
            prev_close = candles[i - 1].close
            tr = max(
                candle.high - candle.low,
                abs(candle.high - prev_close),
                abs(candle.low - prev_close),
            )
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


def fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def init_trade_csv(path: str) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "symbol",
                "direction",
                "entry_price",
                "exit_price",
                "quantity",
                "gross_pnl",
                "commission",
                "net_pnl",
                "balance",
                "exit_reason",
            ]
        )


def append_trade_csv(path: str, trade: Trade) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                fmt_ts(trade.timestamp),
                trade.symbol,
                trade.direction,
                f"{trade.entry_price:.8f}",
                f"{trade.exit_price:.8f}",
                f"{trade.quantity:.8f}",
                f"{trade.gross_pnl:.8f}",
                f"{trade.fees:.8f}",
                f"{trade.net_pnl:.8f}",
                f"{trade.balance:.8f}",
                trade.exit_reason,
            ]
        )


def init_summary_csv(path: str) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "pnl_total", "pnl_last_hour", "total_trades", "wins", "losses", "balance"])


def append_summary_csv(path: str, summary: HourlySummary) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                fmt_ts(summary.timestamp),
                f"{summary.pnl_total:.8f}",
                f"{summary.pnl_last_hour:.8f}",
                summary.total_trades,
                summary.wins,
                summary.losses,
                f"{summary.balance:.8f}",
            ]
        )


# -----------------------------
# Strategy logic
# -----------------------------
def generate_entry_signal(cfg: Config, candles: list[Candle], spread: float) -> tuple[str | None, float | None]:
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    ema_fast_values = ema(closes, cfg.ema_fast)
    ema_trend_values = ema(closes, cfg.ema_trend)
    rsi_values = rsi(closes, cfg.rsi_period)
    atr_values = atr(candles, cfg.atr_period)
    vol_ma_values = sma(volumes, cfg.volume_ma_period)

    i = len(candles) - 1
    c = candles[i]
    prev = candles[i - 1]

    e20 = ema_fast_values[i]
    e50 = ema_trend_values[i]
    r = rsi_values[i]
    a = atr_values[i]
    vma = vol_ma_values[i]
    prev_e20 = ema_fast_values[i - 1]

    if any(v is None for v in [e20, e50, r, a, vma, prev_e20]):
        return None, None

    volume_ok = c.volume > float(vma)
    spread_ok = spread <= cfg.max_spread
    if not volume_ok or not spread_ok:
        return None, None

    long_trend = c.close > float(e50)
    short_trend = c.close < float(e50)
    long_trigger = prev.close <= float(prev_e20) and c.close > float(e20)
    short_trigger = prev.close >= float(prev_e20) and c.close < float(e20)

    long_rsi_ok = (float(r) > cfg.rsi_threshold) if cfg.use_rsi_filter else True
    short_rsi_ok = (float(r) < cfg.rsi_threshold) if cfg.use_rsi_filter else True

    if long_trend and long_trigger and long_rsi_ok:
        return "LONG", float(a)
    if short_trend and short_trigger and short_rsi_ok:
        return "SHORT", float(a)

    return None, None


def evaluate_position_exit(position: Position, candle: Candle, fee_rate: float, balance: float) -> tuple[Trade | None, float]:
    exit_price: float | None = None
    reason = ""

    if position.direction == "LONG":
        sl_hit = candle.low <= position.stop_loss
        tp_hit = candle.high >= position.take_profit
        if sl_hit and tp_hit:
            exit_price = position.stop_loss
            reason = "SL_and_TP_same_candle_assume_SL"
        elif sl_hit:
            exit_price = position.stop_loss
            reason = "SL"
        elif tp_hit:
            exit_price = position.take_profit
            reason = "TP"
    else:
        sl_hit = candle.high >= position.stop_loss
        tp_hit = candle.low <= position.take_profit
        if sl_hit and tp_hit:
            exit_price = position.stop_loss
            reason = "SL_and_TP_same_candle_assume_SL"
        elif sl_hit:
            exit_price = position.stop_loss
            reason = "SL"
        elif tp_hit:
            exit_price = position.take_profit
            reason = "TP"

    if exit_price is None:
        return None, balance

    entry_notional = position.entry_price * position.qty
    exit_notional = exit_price * position.qty
    fees = (entry_notional * fee_rate) + (exit_notional * fee_rate)

    if position.direction == "LONG":
        gross = (exit_price - position.entry_price) * position.qty
    else:
        gross = (position.entry_price - exit_price) * position.qty

    net = gross - fees
    balance += net

    trade = Trade(
        timestamp=candle.close_time,
        symbol=position.symbol,
        direction="شراء" if position.direction == "LONG" else "بيع",
        entry_price=position.entry_price,
        exit_price=exit_price,
        quantity=position.qty,
        gross_pnl=gross,
        fees=fees,
        net_pnl=net,
        balance=balance,
        exit_reason=reason,
    )
    return trade, balance


# -----------------------------
# Live loop
# -----------------------------
def create_open_position(symbol: str, direction: str, atr_value: float, candles: list[Candle], cfg: Config) -> Position | None:
    entry_price = candles[-1].close
    if entry_price <= 0:
        return None

    notional = cfg.margin_per_trade * cfg.leverage
    qty = notional / entry_price
    if qty <= 0:
        return None

    risk = cfg.atr_sl_mult * atr_value
    if risk <= 0:
        return None

    if direction == "LONG":
        stop_loss = entry_price - risk
        take_profit = entry_price + (cfg.rr_ratio * risk)
    else:
        stop_loss = entry_price + risk
        take_profit = entry_price - (cfg.rr_ratio * risk)

    return Position(
        symbol=symbol,
        direction=direction,
        entry_time=candles[-1].close_time,
        entry_price=entry_price,
        qty=qty,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


def send_entry_alert_ar(session: requests.Session, pos: Position, cfg: Config) -> None:
    text = (
        "🟢 فتح صفقة جديدة\n"
        f"رمز العقد: {pos.symbol}\n"
        f"الاتجاه: {'شراء' if pos.direction == 'LONG' else 'بيع'}\n"
        f"سعر الدخول: {pos.entry_price:.4f}\n"
        f"الكمية: {pos.qty:.6f}\n"
        f"وقف الخسارة: {pos.stop_loss:.4f}\n"
        f"جني الأرباح: {pos.take_profit:.4f}\n"
        f"الرافعة: {cfg.leverage:.1f}x\n"
        f"حجم الصفقة (هامش): {cfg.margin_per_trade:.2f} USDT"
    )
    send_telegram_message(session, text)


def send_close_alert_ar(session: requests.Session, trade: Trade) -> None:
    text = (
        "🔴 إغلاق صفقة\n"
        f"رمز العقد: {trade.symbol}\n"
        f"الاتجاه: {trade.direction}\n"
        f"سعر الدخول: {trade.entry_price:.4f}\n"
        f"سعر الخروج: {trade.exit_price:.4f}\n"
        f"الربح/الخسارة الإجمالي: {trade.gross_pnl:.4f} USDT\n"
        f"العمولات: {trade.fees:.4f} USDT\n"
        f"الربح/الخسارة الصافي: {trade.net_pnl:.4f} USDT\n"
        f"الرصيد الحالي: {trade.balance:.4f} USDT\n"
        f"سبب الإغلاق: {trade.exit_reason}"
    )
    send_telegram_message(session, text)


def send_hourly_summary_ar(session: requests.Session, summary: HourlySummary) -> None:
    text = (
        "📊 ملخص الأداء (كل ساعة)\n"
        f"إجمالي الربح/الخسارة منذ التشغيل: {summary.pnl_total:.4f} USDT\n"
        f"الربح/الخسارة خلال آخر ساعة: {summary.pnl_last_hour:.4f} USDT\n"
        f"عدد الصفقات: {summary.total_trades}\n"
        f"الصفقات الرابحة: {summary.wins}\n"
        f"الصفقات الخاسرة: {summary.losses}\n"
        f"الرصيد الحالي: {summary.balance:.4f} USDT"
    )
    send_telegram_message(session, text)


def build_hourly_summary(now_ms: int, trades: list[Trade], balance: float) -> HourlySummary:
    pnl_total = sum(t.net_pnl for t in trades)
    one_hour_ago = now_ms - 3600_000
    pnl_last_hour = sum(t.net_pnl for t in trades if t.timestamp >= one_hour_ago)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    losses = sum(1 for t in trades if t.net_pnl < 0)

    return HourlySummary(
        timestamp=now_ms,
        pnl_total=pnl_total,
        pnl_last_hour=pnl_last_hour,
        total_trades=len(trades),
        wins=wins,
        losses=losses,
        balance=balance,
    )


def run_live_trader(session: requests.Session, cfg: Config) -> None:
    balance = cfg.initial_balance
    trades: list[Trade] = []
    open_position: Position | None = None
    last_summary_sent = 0.0
    loop_count = 0

    init_trade_csv(cfg.trades_csv)
    init_summary_csv(cfg.summary_csv)

    logging.info(
        "Starting live loop | initial_balance=%.2f margin_per_trade=%.2f leverage=%.2fx fee=%.5f",
        cfg.initial_balance,
        cfg.margin_per_trade,
        cfg.leverage,
        cfg.fee_rate,
    )

    while True:
        loop_count += 1
        loop_start = time.time()

        try:
            top_symbols = fetch_top_symbols(session, cfg.top_n_symbols)
            logging.info("Top %d symbols by 24h quote volume: %s", cfg.top_n_symbols, ", ".join(top_symbols))

            # 1) If there is an open position, monitor it first.
            if open_position is not None:
                candles = fetch_klines(session, open_position.symbol, cfg.interval, cfg.candle_limit)
                latest_candle = candles[-1]
                trade, balance = evaluate_position_exit(open_position, latest_candle, cfg.fee_rate, balance)
                if trade is not None:
                    trades.append(trade)
                    append_trade_csv(cfg.trades_csv, trade)
                    logging.info(
                        "Closed %s %s | entry=%.4f exit=%.4f gross=%.4f fee=%.4f net=%.4f balance=%.4f",
                        trade.symbol,
                        trade.direction,
                        trade.entry_price,
                        trade.exit_price,
                        trade.gross_pnl,
                        trade.fees,
                        trade.net_pnl,
                        trade.balance,
                    )
                    if cfg.send_telegram:
                        try:
                            send_close_alert_ar(session, trade)
                        except Exception as exc:  # noqa: BLE001
                            logging.error("Failed to send Arabic close alert: %s", exc)
                    open_position = None

            # 2) If no open position, scan top symbols for first valid opportunity.
            if open_position is None:
                for symbol in top_symbols:
                    try:
                        candles = fetch_klines(session, symbol, cfg.interval, cfg.candle_limit)
                        spread = fetch_spread(session, symbol)
                        signal, atr_value = generate_entry_signal(cfg, candles, spread)

                        if signal is None or atr_value is None:
                            continue

                        pos = create_open_position(symbol, signal, atr_value, candles, cfg)
                        if pos is None:
                            continue

                        open_position = pos
                        logging.info(
                            "Opened %s %s | entry=%.4f qty=%.6f sl=%.4f tp=%.4f spread=%.5f",
                            pos.symbol,
                            pos.direction,
                            pos.entry_price,
                            pos.qty,
                            pos.stop_loss,
                            pos.take_profit,
                            spread,
                        )
                        if cfg.send_telegram:
                            try:
                                send_entry_alert_ar(session, pos, cfg)
                            except Exception as exc:  # noqa: BLE001
                                logging.error("Failed to send Arabic entry alert: %s", exc)
                        break  # strict single-position logic
                    except Exception as exc:  # noqa: BLE001
                        logging.warning("Signal evaluation failed for %s: %s", symbol, exc)

            # 3) Hourly summary
            now = time.time()
            if (now - last_summary_sent) >= cfg.summary_interval_seconds:
                now_ms = int(now * 1000)
                summary = build_hourly_summary(now_ms, trades, balance)
                append_summary_csv(cfg.summary_csv, summary)
                logging.info(
                    "Hourly summary | pnl_total=%.4f pnl_last_hour=%.4f trades=%d wins=%d losses=%d balance=%.4f",
                    summary.pnl_total,
                    summary.pnl_last_hour,
                    summary.total_trades,
                    summary.wins,
                    summary.losses,
                    summary.balance,
                )
                if cfg.send_telegram:
                    try:
                        send_hourly_summary_ar(session, summary)
                    except Exception as exc:  # noqa: BLE001
                        logging.error("Failed to send Arabic hourly summary: %s", exc)
                last_summary_sent = now

        except Exception as exc:  # noqa: BLE001
            logging.error("Main loop error: %s", exc)

        if cfg.max_loops is not None and loop_count >= cfg.max_loops:
            logging.info("Reached max_loops=%d, exiting.", cfg.max_loops)
            break

        elapsed = time.time() - loop_start
        sleep_time = max(1.0, cfg.loop_sleep_seconds - elapsed)
        time.sleep(sleep_time)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Binance futures trend-pullback trader with Arabic Telegram alerts")
    parser.add_argument("--interval", type=str)
    parser.add_argument("--candle-limit", type=int)
    parser.add_argument("--top-n-symbols", type=int)
    parser.add_argument("--fee-rate", type=float)
    parser.add_argument("--leverage", type=float)
    parser.add_argument("--margin-per-trade", type=float)
    parser.add_argument("--max-spread", type=float)
    parser.add_argument("--summary-interval-seconds", type=int)
    parser.add_argument("--loop-sleep-seconds", type=int)
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument("--max-loops", type=int, help="Optional testing cap for main loop iterations")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = load_config_from_env()

    if args.interval:
        cfg.interval = args.interval
    if args.candle_limit is not None:
        cfg.candle_limit = args.candle_limit
    if args.top_n_symbols is not None:
        cfg.top_n_symbols = args.top_n_symbols
    if args.fee_rate is not None:
        cfg.fee_rate = args.fee_rate
    if args.leverage is not None:
        cfg.leverage = args.leverage
    if args.margin_per_trade is not None:
        cfg.margin_per_trade = args.margin_per_trade
    if args.max_spread is not None:
        cfg.max_spread = args.max_spread
    if args.summary_interval_seconds is not None:
        cfg.summary_interval_seconds = args.summary_interval_seconds
    if args.loop_sleep_seconds is not None:
        cfg.loop_sleep_seconds = args.loop_sleep_seconds
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
        cfg = build_config(parse_args())
        with requests.Session() as session:
            run_live_trader(session, cfg)
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
