"""
AlphaBot — Continuous 24/7 Scalping Bot
- TP/SL exit checks every 15 seconds
- New signal scan + trade every 2 minutes using 1-minute bars
- All config live from settings.json
"""

import os, json, time, logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
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

DEFAULT_SETTINGS = {
    "max_capital": 5000,
    "position_size": 100,
    "take_profit_pct": 2.0,
    "stop_loss_pct": 1.0,
    "scan_interval_min": 2,
    "long_threshold": 4,
    "short_threshold": 1,
    "enable_stocks": True,
    "enable_crypto": True,
    "enable_memecoins": True,
    "enable_shorts": True,
    "eod_close_stocks": True,
    "eod_time": "20:45",
    "dashboard_user": "lennert",
    "dashboard_password": "alphabot2024",
    "stock_universe": [
        "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD",
        "JPM","BAC","GS","MS","C","V","MA","PYPL",
        "XOM","CVX","OXY","UNH","PFE","MRNA","ABBV",
        "NFLX","DIS","NKE","SBUX",
        "SPY","QQQ","IWM","XLF","XLK","XLE","ARKK","SOXL",
        "COIN","MSTR","PLTR","RBLX","SNAP","UBER","RIVN",
        "HOOD","SOFI","AFRM","DKNG"
    ],
    "crypto_universe": [
        "BTC/USD","ETH/USD","SOL/USD","AVAX/USD","LINK/USD",
        "LTC/USD","BCH/USD","XRP/USD","UNI/USD","AAVE/USD",
        "DOT/USD","MATIC/USD","DOGE/USD","SHIB/USD"
    ]
}

def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()
    with open(SETTINGS_FILE) as f:
        data = json.load(f)
    updated = False
    for k, v in DEFAULT_SETTINGS.items():
        if k not in data:
            data[k] = v
            updated = True
    if updated:
        save_settings(data)
    return data

def save_settings(data: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Settings saved")

def cfg(key):
    return load_settings()[key]

# ── Clients ───────────────────────────────────────────────────────────────────

trading_client     = TradingClient(API_KEY, SECRET_KEY, paper=True)
stock_data_client  = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data_client = CryptoHistoricalDataClient()

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_crypto(symbol: str) -> bool:
    return "/" in symbol

def get_universe() -> list[str]:
    s = load_settings()
    stocks  = s["stock_universe"]  if s["enable_stocks"]  else []
    cryptos = s["crypto_universe"] if s["enable_crypto"]  else []
    return stocks + cryptos

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

# ── Bar fetching — 1-minute bars for fast signals ─────────────────────────────

def fetch_bars(symbol: str) -> pd.DataFrame | None:
    """
    Fetch 1-minute bars for the last 3 hours (intraday) for crypto,
    or last 2 trading days for stocks (so we have enough bars for indicators).
    Falls back to daily bars if intraday not available.
    """
    try:
        start = datetime.utcnow() - timedelta(hours=6)
        if is_crypto(symbol):
            req  = CryptoBarsRequest(symbol_or_symbols=symbol,
                                     timeframe=TimeFrame.Minute, start=start)
            bars = crypto_data_client.get_crypto_bars(req).df
        else:
            req  = StockBarsRequest(symbol_or_symbols=symbol,
                                    timeframe=TimeFrame.Minute, start=start)
            bars = stock_data_client.get_stock_bars(req).df

        if bars is None or bars.empty:
            return None
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")
        bars = bars.sort_index()
        # Need at least 30 bars for indicators
        return bars if len(bars) >= 30 else None
    except Exception as e:
        log.debug(f"fetch_bars({symbol}): {e}")
        return None

# ── Scoring on 1-minute bars ──────────────────────────────────────────────────

def score_symbol(symbol: str) -> tuple[float, dict]:
    s    = load_settings()
    bars = fetch_bars(symbol)
    if bars is None:
        return 3.0, {}

    close  = bars["close"]
    volume = bars["volume"]
    signals = {}
    score   = 0.0

    # 1. Short momentum: last 5 bars (5 min)
    ret_5   = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0
    signals["ret_5d"] = round(ret_5 * 100, 3)
    if ret_5 > 0: score += 1

    # 2. Medium momentum: last 30 bars (30 min)
    ret_30  = (close.iloc[-1] / close.iloc[-31] - 1) if len(close) >= 31 else 0
    signals["ret_20d"] = round(ret_30 * 100, 3)
    if ret_30 > 0.002: score += 1   # 0.2% in 30 min

    # 3. RSI(14)
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    signals["rsi"] = round(rsi, 1)
    if 45 < rsi < 70: score += 1
    elif rsi < 30:    score -= 1

    # 4. MACD
    macd_obj  = ta.trend.MACD(close)
    macd_line = macd_obj.macd().iloc[-1]
    macd_sig  = macd_obj.macd_signal().iloc[-1]
    signals["macd_above_signal"] = bool(macd_line > macd_sig)
    if macd_line > macd_sig: score += 1

    # 5. Price above 20-bar SMA
    sma20 = close.rolling(20).mean().iloc[-1]
    signals["above_sma20"] = bool(close.iloc[-1] > sma20)
    if close.iloc[-1] > sma20: score += 1

    # 6. Volume spike vs 20-bar avg
    vol_avg   = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_avg if vol_avg else 1
    signals["vol_ratio"] = round(vol_ratio, 2)
    if vol_ratio > 1.5: score += 1

    return max(0.0, min(6.0, score)), signals

# ── TP/SL exits — runs every 15 seconds ──────────────────────────────────────

def check_exits():
    s  = load_settings()
    tp = s["take_profit_pct"] / 100
    sl = s["stop_loss_pct"]   / 100
    closed = 0
    for sym, pos in get_positions().items():
        plpc = float(pos.unrealized_plpc)
        if plpc >= tp:
            log.info(f"  ✅ TAKE PROFIT → {sym} @ {plpc*100:+.2f}%")
            try: trading_client.close_position(sym); closed += 1
            except Exception as e: log.error(f"  Close failed {sym}: {e}")
        elif plpc <= -sl:
            log.info(f"  🛑 STOP LOSS   → {sym} @ {plpc*100:+.2f}%")
            try: trading_client.close_position(sym); closed += 1
            except Exception as e: log.error(f"  Close failed {sym}: {e}")
    return closed

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

# ── Full scan + trade ─────────────────────────────────────────────────────────

def scan_and_trade():
    s = load_settings()
    log.info(f"── SCAN {datetime.utcnow().strftime('%H:%M:%S')} UTC ──")

    account   = get_account()
    cash      = float(account.cash)
    positions = get_positions()
    deployed  = sum(abs(float(p.market_value)) for p in positions.values())
    remaining = max(0.0, s["max_capital"] - deployed)
    pos_size  = s["position_size"]

    if remaining < pos_size:
        log.info(f"Cap full — deployed ${deployed:,.0f} / ${s['max_capital']:,.0f}")
        return

    market_open = is_market_open()
    longs, shorts = [], []

    for sym in get_universe():
        if sym in positions:
            continue
        if not is_crypto(sym) and not market_open:
            continue
        score, signals = score_symbol(sym)
        if score >= s["long_threshold"]:
            longs.append((sym, score, signals))
        elif score <= s["short_threshold"] and s["enable_shorts"]:
            shorts.append((sym, score, signals))

    longs.sort(key=lambda x: x[1], reverse=True)
    shorts.sort(key=lambda x: x[1])

    candidates = (
        [(sym, sc, sig, OrderSide.BUY)  for sym, sc, sig in longs] +
        [(sym, sc, sig, OrderSide.SELL) for sym, sc, sig in shorts]
    )

    if candidates:
        log.info(f"  Signals: {len(longs)}▲ {len(shorts)}▼ | Remaining: ${remaining:,.0f}")

    for sym, score, signals, side in candidates:
        if remaining < pos_size or cash < pos_size:
            break
        place_order(sym, pos_size, side)
        remaining -= pos_size
        cash      -= pos_size
        time.sleep(0.2)

# ── EOD ───────────────────────────────────────────────────────────────────────

def end_of_day_run():
    s = load_settings()
    if not s["eod_close_stocks"]:
        return
    log.info("EOD — closing stock positions")
    for sym, pos in get_positions().items():
        if is_crypto(sym):
            continue
        try:
            trading_client.close_position(sym)
            log.info(f"  Closed {sym}: P/L={float(pos.unrealized_plpc)*100:+.1f}%")
        except Exception as e:
            log.error(f"  Failed {sym}: {e}")

def print_portfolio_summary():
    account   = get_account()
    positions = get_positions()
    total_pl  = sum(float(p.unrealized_pl) for p in positions.values())
    log.info(f"SUMMARY | Equity: ${float(account.equity):,.2f} | "
             f"Positions: {len(positions)} | P/L: ${total_pl:+,.2f}")
