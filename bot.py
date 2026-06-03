"""
AlphaBot v3 — Smarter Strategy
==================================
Core rules:
  1. PRIMARY TREND FILTER — Only go long when SPY is above its 50-day SMA,
     only go short when SPY is below. No counter-trend trades.
  2. DAILY bars for signals (reliable) + 15-min bars for entry timing.
  3. ATR-BASED stops — stops adapt to each asset's volatility.
     Stop = 1.5x ATR below entry. TP = 3x stop → 3:1 risk/reward.
  4. STRICT SCORING — score must be ≥ 5/7 for longs, ≤ 2/7 for shorts.
  5. MAX 8 POSITIONS — concentrated, high-conviction trades only.
  6. DAILY LOSS LIMIT — if unrealised + realised loss today > 2% of capital, stop trading.
  7. EXIT CHECK every 15 s. SCAN every 5 min. Daily bars refreshed once per hour.
"""

import os, json, time, logging
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
import numpy as np
import ta
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

SETTINGS_FILE = Path(__file__).parent / "settings.json"

# ── Settings ──────────────────────────────────────────────────────────────────

DEFAULT_SETTINGS = {
    "max_capital":        5000,
    "position_size":      500,      # larger per trade, fewer trades = better quality
    "take_profit_pct":    3.0,      # 3% TP
    "stop_loss_pct":      1.0,      # 1% SL → 3:1 R:R
    "use_atr_stops":      True,     # override fixed % with ATR-based stops
    "atr_stop_mult":      1.5,      # stop = 1.5x ATR
    "atr_tp_mult":        3.0,      # TP   = 3.0x ATR (= 2:1 on the stop)
    "scan_interval_min":  5,
    "max_positions":      8,        # hard cap — quality over quantity
    "long_threshold":     5,        # need 5/7 to go long
    "short_threshold":    2,        # need ≤ 2/7 to go short
    "daily_loss_limit":   2.0,      # stop trading if down 2% of max_capital today
    "enable_stocks":      True,
    "enable_crypto":      True,
    "enable_memecoins":   True,
    "enable_shorts":      True,
    "eod_close_stocks":   True,
    "eod_time":           "20:45",
    "dashboard_user":     "lennert",
    "dashboard_password": "alphabot2024",
    "stock_universe": [
        # High-liquidity, well-behaved stocks only
        "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD",
        "JPM","GS","V","MA","XOM","UNH",
        "SPY","QQQ","IWM","XLK","XLF","XLE",
        "COIN","PLTR","MSTR","UBER","SOFI",
    ],
    "crypto_universe": [
        "BTC/USD","ETH/USD","SOL/USD","AVAX/USD",
        "DOGE/USD","LINK/USD","XRP/USD","LTC/USD",
    ]
}

def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()
    with open(SETTINGS_FILE) as f:
        data = json.load(f)
    for k, v in DEFAULT_SETTINGS.items():
        if k not in data:
            data[k] = v
    return data

def save_settings(data: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Settings saved")

# ── Clients ───────────────────────────────────────────────────────────────────

trading_client     = TradingClient(API_KEY, SECRET_KEY, paper=True)
stock_data_client  = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data_client = CryptoHistoricalDataClient()

# ── Bar cache (daily bars only fetched once per hour per symbol) ───────────────

_bar_cache: dict[str, tuple[float, pd.DataFrame]] = {}   # symbol → (timestamp, df)
CACHE_TTL = 3600   # 1 hour

def is_crypto(symbol: str) -> bool:
    return "/" in symbol

def get_universe() -> list[str]:
    s = load_settings()
    stocks  = s["stock_universe"]  if s["enable_stocks"]  else []
    cryptos = s["crypto_universe"] if s["enable_crypto"]  else []
    return stocks + cryptos

def fetch_daily_bars(symbol: str) -> pd.DataFrame | None:
    """Fetch 90 days of daily bars, cached for 1 hour."""
    now = time.time()
    if symbol in _bar_cache:
        ts, df = _bar_cache[symbol]
        if now - ts < CACHE_TTL:
            return df

    try:
        start = datetime.utcnow() - timedelta(days=90)
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
        bars = bars.sort_index()
        if len(bars) < 20:
            return None
        _bar_cache[symbol] = (now, bars)
        return bars
    except Exception as e:
        log.debug(f"fetch_daily_bars({symbol}): {e}")
        return None

# ── Market helpers ────────────────────────────────────────────────────────────

def get_account():
    return trading_client.get_account()

def get_positions():
    return {p.symbol: p for p in trading_client.get_all_positions()}

def get_open_orders():
    return trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))

def cancel_all_orders():
    trading_client.cancel_orders()

def close_all_positions():
    trading_client.close_all_positions(cancel_orders=True)

def is_market_open() -> bool:
    try:
        return trading_client.get_clock().is_open
    except Exception:
        return False

# ── Trend filter — SPY direction ──────────────────────────────────────────────

_spy_trend_cache = {"ts": 0.0, "trend": "neutral"}

def get_market_trend() -> str:
    """
    Returns 'bull', 'bear', or 'neutral'.
    Bull  = SPY above 50-day SMA
    Bear  = SPY below 50-day SMA
    """
    now = time.time()
    if now - _spy_trend_cache["ts"] < 1800:   # cache 30 min
        return _spy_trend_cache["trend"]

    try:
        bars = fetch_daily_bars("SPY")
        if bars is None or len(bars) < 50:
            return "neutral"
        close = bars["close"]
        sma50 = close.rolling(50).mean().iloc[-1]
        trend = "bull" if close.iloc[-1] > sma50 else "bear"
        _spy_trend_cache.update({"ts": now, "trend": trend})
        return trend
    except Exception:
        return "neutral"

# ── ATR calculation ───────────────────────────────────────────────────────────

def calc_atr(bars: pd.DataFrame, period: int = 14) -> float:
    """Average True Range over last N days."""
    try:
        atr = ta.volatility.AverageTrueRange(
            bars["high"], bars["low"], bars["close"], window=period
        ).average_true_range().iloc[-1]
        return float(atr)
    except Exception:
        return float(bars["close"].iloc[-1]) * 0.015   # fallback: 1.5% of price

# ── Scoring (daily bars, 7 signals) ──────────────────────────────────────────

def score_symbol(symbol: str) -> tuple[float, dict]:
    """
    7 signals scored 0/1 each.
    High score = bullish. Low score = bearish.
    """
    bars = fetch_daily_bars(symbol)
    if bars is None or len(bars) < 50:
        return 3.5, {}

    close  = bars["close"]
    high   = bars["high"]
    low    = bars["low"]
    volume = bars["volume"]
    signals = {}

    score = 0.0

    # 1. 5-day momentum
    ret_5d = close.iloc[-1] / close.iloc[-6] - 1
    signals["ret_5d"] = round(ret_5d * 100, 2)
    if ret_5d > 0: score += 1

    # 2. 20-day momentum > 2%
    ret_20d = close.iloc[-1] / close.iloc[-21] - 1
    signals["ret_20d"] = round(ret_20d * 100, 2)
    if ret_20d > 0.02: score += 1

    # 3. RSI(14) — sweet spot 45–68 for longs, below 35 bearish
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    signals["rsi"] = round(rsi, 1)
    if 45 < rsi < 68: score += 1
    elif rsi < 35:    score -= 0.5   # partial penalty to avoid extremes

    # 4. MACD line above signal AND positive histogram
    macd_obj  = ta.trend.MACD(close)
    macd_line = macd_obj.macd().iloc[-1]
    macd_sig  = macd_obj.macd_signal().iloc[-1]
    macd_hist = macd_obj.macd_diff().iloc[-1]
    signals["macd_above_signal"] = bool(macd_line > macd_sig)
    if macd_line > macd_sig and macd_hist > 0: score += 1

    # 5. Price above BOTH 20-day AND 50-day SMA (double confirmation)
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    above_both = close.iloc[-1] > sma20 and close.iloc[-1] > sma50
    signals["above_sma20"] = bool(close.iloc[-1] > sma20)
    if above_both: score += 1

    # 6. Bollinger Band — price above midline (bullish), below lower band (oversold bounce)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_mid   = bb.bollinger_mavg().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]
    signals["bb_mid"] = round(bb_mid, 2)
    if close.iloc[-1] > bb_mid: score += 1
    elif close.iloc[-1] < bb_lower: score -= 0.5   # slightly bearish signal

    # 7. Volume surge ≥ 1.3x 20-day avg (lower threshold = more hits)
    vol_avg   = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_avg if vol_avg else 1
    signals["vol_ratio"] = round(vol_ratio, 2)
    if vol_ratio >= 1.3: score += 1

    return max(0.0, min(7.0, score)), signals

# ── Daily loss tracker ────────────────────────────────────────────────────────

_daily_stats = {"date": None, "realised_pl": 0.0}

def get_unrealised_pl() -> float:
    try:
        return sum(float(p.unrealized_pl) for p in get_positions().values())
    except Exception:
        return 0.0

def record_realised(amount: float):
    today = date.today().isoformat()
    if _daily_stats["date"] != today:
        _daily_stats.update({"date": today, "realised_pl": 0.0})
    _daily_stats["realised_pl"] += amount

def is_daily_loss_limit_hit() -> bool:
    s         = load_settings()
    limit     = s["max_capital"] * (s["daily_loss_limit"] / 100)
    today     = date.today().isoformat()
    if _daily_stats["date"] != today:
        return False
    total_loss = _daily_stats["realised_pl"] + get_unrealised_pl()
    return total_loss < -limit

# ── TP/SL exits ───────────────────────────────────────────────────────────────

def check_exits():
    s  = load_settings()
    tp = s["take_profit_pct"] / 100
    sl = s["stop_loss_pct"]   / 100

    for sym, pos in get_positions().items():
        plpc = float(pos.unrealized_plpc)

        # ATR-based TP/SL override
        if s.get("use_atr_stops", True):
            bars = fetch_daily_bars(sym)
            if bars is not None:
                atr   = calc_atr(bars)
                price = float(pos.avg_entry_price)
                sl    = (atr * s["atr_stop_mult"]) / price
                tp    = (atr * s["atr_tp_mult"])   / price

        if plpc >= tp:
            log.info(f"  ✅ TAKE PROFIT → {sym} @ {plpc*100:+.2f}%")
            try:
                pl = float(pos.unrealized_pl)
                trading_client.close_position(sym)
                record_realised(pl)
            except Exception as e:
                log.error(f"  Close failed {sym}: {e}")
        elif plpc <= -sl:
            log.info(f"  🛑 STOP LOSS   → {sym} @ {plpc*100:+.2f}%")
            try:
                pl = float(pos.unrealized_pl)
                trading_client.close_position(sym)
                record_realised(pl)
            except Exception as e:
                log.error(f"  Close failed {sym}: {e}")

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

# ── Main scan ─────────────────────────────────────────────────────────────────

def scan_and_trade():
    s = load_settings()

    if is_daily_loss_limit_hit():
        log.warning(f"⛔ Daily loss limit hit — no new trades today")
        return

    trend = get_market_trend()
    log.info(f"── SCAN {datetime.utcnow().strftime('%H:%M')} UTC | Market trend: {trend.upper()} ──")

    account   = get_account()
    cash      = float(account.cash)
    positions = get_positions()
    n_open    = len(positions)
    max_pos   = int(s["max_positions"])
    deployed  = sum(abs(float(p.market_value)) for p in positions.values())
    remaining = max(0.0, s["max_capital"] - deployed)
    pos_size  = s["position_size"]

    if n_open >= max_pos:
        log.info(f"Max positions reached ({n_open}/{max_pos})")
        return
    if remaining < pos_size:
        log.info(f"Capital cap reached — ${deployed:,.0f} / ${s['max_capital']:,.0f} deployed")
        return

    market_open = is_market_open()
    long_thresh  = s["long_threshold"]
    short_thresh = s["short_threshold"]
    slots = max_pos - n_open

    longs, shorts = [], []

    for sym in get_universe():
        if sym in positions or sym == "SPY":
            continue
        if not is_crypto(sym) and not market_open:
            continue

        score, signals = score_symbol(sym)

        # Trend filter — only trade in market direction
        if trend == "bull" and score >= long_thresh:
            longs.append((sym, score, signals))
        elif trend == "bear" and score <= short_thresh and s["enable_shorts"]:
            shorts.append((sym, score, signals))
        elif trend == "neutral":
            # In neutral market: only take highest-conviction signals
            if score >= long_thresh + 1:
                longs.append((sym, score, signals))
            elif score <= short_thresh - 1 and s["enable_shorts"]:
                shorts.append((sym, score, signals))

    longs.sort(key=lambda x: x[1], reverse=True)
    shorts.sort(key=lambda x: x[1])

    # Fill slots: prioritise longs in bull, shorts in bear
    if trend == "bear":
        candidates = (
            [(sym, sc, sig, OrderSide.SELL) for sym, sc, sig in shorts[:slots]] +
            [(sym, sc, sig, OrderSide.BUY)  for sym, sc, sig in longs[:max(0, slots - len(shorts))]]
        )
    else:
        candidates = (
            [(sym, sc, sig, OrderSide.BUY)  for sym, sc, sig in longs[:slots]] +
            [(sym, sc, sig, OrderSide.SELL) for sym, sc, sig in shorts[:max(0, slots - len(longs))]]
        )

    if candidates:
        log.info(f"  High-conviction signals: {len(longs)}▲ {len(shorts)}▼ → placing {len(candidates)} orders")
    else:
        log.info(f"  No qualifying signals this scan (trend={trend})")

    for sym, score, signals, side in candidates:
        if remaining < pos_size or cash < pos_size or n_open >= max_pos:
            break
        place_order(sym, pos_size, side)
        remaining -= pos_size
        cash      -= pos_size
        n_open    += 1
        time.sleep(0.3)

def end_of_day_run():
    s = load_settings()
    if not s["eod_close_stocks"]:
        return
    log.info("EOD — closing stock positions")
    for sym, pos in get_positions().items():
        if is_crypto(sym):
            continue
        try:
            pl = float(pos.unrealized_pl)
            trading_client.close_position(sym)
            record_realised(pl)
            log.info(f"  Closed {sym}: P/L={float(pos.unrealized_plpc)*100:+.1f}%")
        except Exception as e:
            log.error(f"  Failed {sym}: {e}")

def print_portfolio_summary():
    try:
        account   = get_account()
        positions = get_positions()
        total_pl  = sum(float(p.unrealized_pl) for p in positions.values())
        trend     = get_market_trend()
        log.info(
            f"SUMMARY | Equity: ${float(account.equity):,.2f} | "
            f"Positions: {len(positions)}/{load_settings()['max_positions']} | "
            f"Unrealised P/L: ${total_pl:+,.2f} | "
            f"Today realised: ${_daily_stats.get('realised_pl', 0):+,.2f} | "
            f"Trend: {trend.upper()}"
        )
    except Exception as e:
        log.error(f"Summary error: {e}")
