"""
AlphaBot — 24/7 Scalping Bot
- Stocks (NYSE hours only) + Crypto + Memecoins (24/7)
- Scans every 30 min, no position count cap, $5K capital cap
- $100/trade, +2% take profit, -1% stop loss
- Goes LONG on score ≥ 4, SHORT on score ≤ 1
"""

import os
import time
import logging
from datetime import datetime, timedelta
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

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

MAX_CAPITAL     = float(os.getenv("MAX_CAPITAL", 5000))
POSITION_SIZE   = 100      # $100 per trade
TAKE_PROFIT_PCT = 0.02     # +2% → close
STOP_LOSS_PCT   = 0.01     # -1% → close
LONG_THRESHOLD  = 4        # score ≥ 4 → long
SHORT_THRESHOLD = 1        # score ≤ 1 → short

# ── Universes ─────────────────────────────────────────────────────────────────

STOCK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "JPM", "BAC", "GS", "MS", "C", "V", "MA", "PYPL",
    "XOM", "CVX", "OXY",
    "UNH", "PFE", "MRNA", "ABBV",
    "NFLX", "DIS", "NKE", "SBUX",
    "SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "ARKK", "SOXL",
    "COIN", "MSTR", "PLTR", "RBLX", "SNAP", "UBER", "RIVN",
    "HOOD", "SOFI", "AFRM", "DKNG",
]

# Alpaca crypto symbols — format is BASE/USD
CRYPTO_UNIVERSE = [
    # Blue-chip crypto
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD",
    "LTC/USD", "BCH/USD", "XRP/USD", "UNI/USD", "AAVE/USD",
    "DOT/USD", "MATIC/USD",
    # Memecoins
    "DOGE/USD", "SHIB/USD",
]

UNIVERSE = STOCK_UNIVERSE + CRYPTO_UNIVERSE

# ── Clients ───────────────────────────────────────────────────────────────────

trading_client     = TradingClient(API_KEY, SECRET_KEY, paper=True)
stock_data_client  = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data_client = CryptoHistoricalDataClient()   # no auth needed for historical

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_crypto(symbol: str) -> bool:
    return "/" in symbol

def get_account():
    return trading_client.get_account()

def get_positions():
    return {p.symbol: p for p in trading_client.get_all_positions()}

def get_open_orders():
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    return trading_client.get_orders(filter=req)

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
    bars = fetch_bars(symbol)
    if bars is None or len(bars) < 20:
        return 3.0, {}

    close  = bars["close"]
    volume = bars["volume"]
    signals = {}
    score   = 0.0

    ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0
    signals["ret_5d"] = round(ret_5d * 100, 2)
    if ret_5d > 0:
        score += 1

    ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0
    signals["ret_20d"] = round(ret_20d * 100, 2)
    if ret_20d > 0.03:
        score += 1

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    signals["rsi"] = round(rsi, 1)
    if 45 < rsi < 70:
        score += 1
    elif rsi < 30:
        score -= 1

    macd_obj  = ta.trend.MACD(close)
    macd_line = macd_obj.macd().iloc[-1]
    macd_sig  = macd_obj.macd_signal().iloc[-1]
    signals["macd_above_signal"] = bool(macd_line > macd_sig)
    if macd_line > macd_sig:
        score += 1

    sma20 = close.rolling(20).mean().iloc[-1]
    signals["above_sma20"] = bool(close.iloc[-1] > sma20)
    if close.iloc[-1] > sma20:
        score += 1

    vol_avg   = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_avg if vol_avg else 1
    signals["vol_ratio"] = round(vol_ratio, 2)
    if vol_ratio > 1.5:
        score += 1

    return max(0.0, min(6.0, score)), signals

# ── TP/SL checker ─────────────────────────────────────────────────────────────

def check_exits():
    for sym, pos in get_positions().items():
        plpc = float(pos.unrealized_plpc)
        if plpc >= TAKE_PROFIT_PCT or plpc <= -STOP_LOSS_PCT:
            reason = "TAKE PROFIT" if plpc >= TAKE_PROFIT_PCT else "STOP LOSS"
            log.info(f"  {reason} → closing {sym} @ {plpc*100:+.2f}%")
            try:
                trading_client.close_position(sym)
            except Exception as e:
                log.error(f"  Failed to close {sym}: {e}")

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

def scan_and_trade(asset_filter: str = "all"):
    """
    asset_filter: "all" | "stocks" | "crypto"
    Crypto always tradeable. Stocks only when NYSE is open.
    """
    log.info("=" * 60)
    log.info(f"SCAN [{asset_filter.upper()}] — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    check_exits()

    account   = get_account()
    cash      = float(account.cash)
    positions = get_positions()
    deployed  = sum(abs(float(p.market_value)) for p in positions.values())
    remaining = max(0.0, MAX_CAPITAL - deployed)

    log.info(f"Positions: {len(positions)} | Deployed: ${deployed:,.2f} | Remaining: ${remaining:,.2f}")

    if remaining < POSITION_SIZE:
        log.info("$5K cap fully deployed — waiting for exits")
        return

    market_open = is_market_open()
    longs, shorts = [], []

    for sym in UNIVERSE:
        if sym in positions:
            continue
        if not is_crypto(sym) and not market_open:
            continue          # skip stocks when NYSE is closed
        if asset_filter == "crypto" and not is_crypto(sym):
            continue
        if asset_filter == "stocks" and is_crypto(sym):
            continue

        score, signals = score_symbol(sym)
        entry = (sym, score, signals)
        if score >= LONG_THRESHOLD:
            longs.append(entry)
        elif score <= SHORT_THRESHOLD:
            shorts.append(entry)

    longs.sort(key=lambda x: x[1], reverse=True)
    shorts.sort(key=lambda x: x[1])

    candidates = (
        [(sym, sc, sig, OrderSide.BUY)  for sym, sc, sig in longs] +
        [(sym, sc, sig, OrderSide.SELL) for sym, sc, sig in shorts]
    )

    log.info(f"Signals: {len(longs)} longs + {len(shorts)} shorts")

    for sym, score, signals, side in candidates:
        if remaining < POSITION_SIZE or cash < POSITION_SIZE:
            log.info("Capital cap reached")
            break
        place_order(sym, POSITION_SIZE, side)
        remaining -= POSITION_SIZE
        cash      -= POSITION_SIZE
        time.sleep(0.3)

def end_of_day_run():
    """Close stock positions at NYSE close. Crypto positions stay open."""
    log.info("=" * 60)
    log.info("EOD — closing stock positions (crypto stays open)")
    positions = get_positions()
    for sym, pos in positions.items():
        if is_crypto(sym):
            continue
        plpc = float(pos.unrealized_plpc) * 100
        pl   = float(pos.unrealized_pl)
        log.info(f"  Closing {sym}: P/L=${pl:+.2f} ({plpc:+.1f}%)")
        try:
            trading_client.close_position(sym)
        except Exception as e:
            log.error(f"  Failed: {e}")

def print_portfolio_summary():
    account   = get_account()
    positions = get_positions()
    stocks    = [(s, p) for s, p in positions.items() if not is_crypto(s)]
    cryptos   = [(s, p) for s, p in positions.items() if is_crypto(s)]
    total_pl  = sum(float(p.unrealized_pl) for p in positions.values())

    log.info("\n--- PORTFOLIO SUMMARY ---")
    log.info(f"Equity:        ${float(account.equity):>12,.2f}")
    log.info(f"Open:          {len(positions)} ({len(stocks)} stocks, {len(cryptos)} crypto)")
    log.info(f"Total P/L:     ${total_pl:>+12,.2f}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan",     action="store_true")
    parser.add_argument("--eod",      action="store_true")
    parser.add_argument("--summary",  action="store_true")
    parser.add_argument("--schedule", action="store_true")
    args = parser.parse_args()

    if not API_KEY or not SECRET_KEY:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        exit(1)

    if args.scan:
        scan_and_trade()
    elif args.eod:
        end_of_day_run()
    elif args.summary:
        print_portfolio_summary()
    elif args.schedule:
        import schedule as sched

        # Every 30 min around the clock (crypto 24/7, stocks gated inside scan_and_trade)
        sched.every(30).minutes.do(scan_and_trade)

        # EOD: close stock positions only, keep crypto running
        for day in ["monday","tuesday","wednesday","thursday","friday"]:
            getattr(sched.every(), day).at("20:45").do(end_of_day_run)

        sched.every().hour.do(print_portfolio_summary)

        log.info("24/7 scheduler started — crypto always, stocks during NYSE hours")
        while True:
            sched.run_pending()
            time.sleep(30)
    else:
        parser.print_help()
