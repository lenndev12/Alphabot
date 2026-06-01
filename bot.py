"""
AlphaBot — 24/7 Scalping Bot
All config is loaded live from settings.json — no restart needed.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import ta
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, PositionSide
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

SETTINGS_FILE = Path(__file__).parent / "settings.json"

# ── Settings ──────────────────────────────────────────────────────────────────

def load_settings() -> dict:
    with open(SETTINGS_FILE) as f:
        return json.load(f)

def save_settings(data: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Settings saved")

def cfg(key):
    """Read a single setting live from disk."""
    return load_settings()[key]

# Expose UNIVERSE as a dynamic property used by the API scan
@property
def UNIVERSE():
    s = load_settings()
    stocks  = s["stock_universe"]  if s["enable_stocks"]     else []
    cryptos = s["crypto_universe"] if s["enable_crypto"]      else []
    return stocks + cryptos

# ── Clients ───────────────────────────────────────────────────────────────────

trading_client     = TradingClient(API_KEY, SECRET_KEY, paper=True)
stock_data_client  = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data_client = CryptoHistoricalDataClient()

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_universe() -> list[str]:
    s = load_settings()
    stocks  = s["stock_universe"]  if s["enable_stocks"]  else []
    cryptos = s["crypto_universe"] if s["enable_crypto"]  else []
    return stocks + cryptos

def is_crypto(symbol: str) -> bool:
    return "/" in symbol

def get_account():
    return trading_client.get_account()

def get_positions():
    return {p.symbol: p for p in trading_client.get_all_positions()}

def get_open_orders():
    return trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))

def cancel_all_orders():
    trading_client.cancel_orders()
    log.info("Cancelled all open orders")

def close_all_positions():
    trading_client.close_all_positions(cancel_orders=True)
    log.info("Closed all positions")

def is_market_open() -> bool:
    try:
        return trading_client.get_clock().is_open
    except Exception:
        return False

def fetch_bars(symbol: str, days: int = 60) -> pd.DataFrame | None:
    try:
        start = datetime.utcnow() - timedelta(days=days)
        if is_crypto(symbol):
            req  = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start)
            bars = crypto_data_client.get_crypto_bars(req).df
        else:
            req  = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start)
            bars = stock_data_client.get_stock_bars(req).df

        if bars is None or bars.empty:
            return None
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")
        return bars.sort_index()
    except Exception as e:
        log.debug(f"fetch_bars({symbol}): {e}")
        return None

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_symbol(symbol: str) -> tuple[float, dict]:
    s     = load_settings()
    bars  = fetch_bars(symbol)
    if bars is None or len(bars) < 20:
        return 3.0, {}

    close  = bars["close"]
    volume = bars["volume"]
    signals = {}
    score   = 0.0

    ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0
    signals["ret_5d"] = round(ret_5d * 100, 2)
    if ret_5d > 0: score += 1

    ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0
    signals["ret_20d"] = round(ret_20d * 100, 2)
    if ret_20d > 0.03: score += 1

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    signals["rsi"] = round(rsi, 1)
    if 45 < rsi < 70: score += 1
    elif rsi < 30:    score -= 1

    macd_obj  = ta.trend.MACD(close)
    macd_line = macd_obj.macd().iloc[-1]
    macd_sig  = macd_obj.macd_signal().iloc[-1]
    signals["macd_above_signal"] = bool(macd_line > macd_sig)
    if macd_line > macd_sig: score += 1

    sma20 = close.rolling(20).mean().iloc[-1]
    signals["above_sma20"] = bool(close.iloc[-1] > sma20)
    if close.iloc[-1] > sma20: score += 1

    vol_avg   = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_avg if vol_avg else 1
    signals["vol_ratio"] = round(vol_ratio, 2)
    if vol_ratio > 1.5: score += 1

    return max(0.0, min(6.0, score)), signals

# ── TP/SL checker ─────────────────────────────────────────────────────────────

def check_exits():
    s  = load_settings()
    tp = s["take_profit_pct"] / 100
    sl = s["stop_loss_pct"]   / 100

    for sym, pos in get_positions().items():
        plpc = float(pos.unrealized_plpc)
        if plpc >= tp:
            log.info(f"  TAKE PROFIT → closing {sym} @ {plpc*100:+.2f}%")
            try: trading_client.close_position(sym)
            except Exception as e: log.error(f"  Failed: {e}")
        elif plpc <= -sl:
            log.info(f"  STOP LOSS   → closing {sym} @ {plpc*100:+.2f}%")
            try: trading_client.close_position(sym)
            except Exception as e: log.error(f"  Failed: {e}")

# ── Order placement ───────────────────────────────────────────────────────────

def place_order(symbol: str, dollar_amount: float, side: OrderSide):
    try:
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(dollar_amount, 2),
            side=side,
            time_in_force=TimeInForce.GTC if is_crypto(symbol) else TimeInForce.DAY,
        )
        order = trading_client.submit_order(req)
        tag = "▲ LONG" if side == OrderSide.BUY else "▼ SHORT"
        log.info(f"  {tag} ${dollar_amount:.0f} {symbol}")
        return order
    except Exception as e:
        log.error(f"  Order failed {symbol}: {e}")
        return None

# ── Core scan ─────────────────────────────────────────────────────────────────

def scan_and_trade():
    s = load_settings()
    log.info("=" * 60)
    log.info(f"SCAN — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    check_exits()

    account   = get_account()
    cash      = float(account.cash)
    positions = get_positions()
    deployed  = sum(abs(float(p.market_value)) for p in positions.values())
    remaining = max(0.0, s["max_capital"] - deployed)
    pos_size  = s["position_size"]

    log.info(f"Positions: {len(positions)} | Deployed: ${deployed:,.2f} | Remaining: ${remaining:,.2f}")

    if remaining < pos_size:
        log.info("Capital cap fully deployed — waiting for exits")
        return

    market_open   = is_market_open()
    long_thresh   = s["long_threshold"]
    short_thresh  = s["short_threshold"]
    shorts_on     = s["enable_shorts"]

    longs, shorts = [], []
    for sym in get_universe():
        if sym in positions:
            continue
        if not is_crypto(sym) and not market_open:
            continue
        score, signals = score_symbol(sym)
        if score >= long_thresh:
            longs.append((sym, score, signals))
        elif score <= short_thresh and shorts_on:
            shorts.append((sym, score, signals))

    longs.sort(key=lambda x: x[1], reverse=True)
    shorts.sort(key=lambda x: x[1])

    candidates = (
        [(sym, sc, sig, OrderSide.BUY)  for sym, sc, sig in longs] +
        [(sym, sc, sig, OrderSide.SELL) for sym, sc, sig in shorts]
    )

    log.info(f"Signals: {len(longs)} longs + {len(shorts)} shorts")

    for sym, score, signals, side in candidates:
        if remaining < pos_size or cash < pos_size:
            log.info("Capital cap reached")
            break
        place_order(sym, pos_size, side)
        remaining -= pos_size
        cash      -= pos_size
        time.sleep(0.3)

def end_of_day_run():
    s = load_settings()
    if not s["eod_close_stocks"]:
        log.info("EOD stock-close disabled in settings — skipping")
        return
    log.info("EOD — closing stock positions (crypto stays open)")
    for sym, pos in get_positions().items():
        if is_crypto(sym):
            continue
        plpc = float(pos.unrealized_plpc) * 100
        log.info(f"  Closing {sym}: P/L={plpc:+.1f}%")
        try: trading_client.close_position(sym)
        except Exception as e: log.error(f"  Failed: {e}")

def print_portfolio_summary():
    account   = get_account()
    positions = get_positions()
    total_pl  = sum(float(p.unrealized_pl) for p in positions.values())
    stocks    = [s for s in positions if not is_crypto(s)]
    cryptos   = [s for s in positions if is_crypto(s)]
    log.info(f"SUMMARY | Equity: ${float(account.equity):,.2f} | "
             f"Positions: {len(positions)} ({len(stocks)} stocks, {len(cryptos)} crypto) | "
             f"P/L: ${total_pl:+,.2f}")
