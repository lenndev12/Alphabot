"""
Flask server with session-based login + bot API.
"""

import threading
import time
import logging
from datetime import datetime
from functools import wraps
from flask import Flask, jsonify, send_from_directory, request, session, redirect, url_for
from flask_cors import CORS
import bot as trading_bot

app = Flask(__name__, static_folder="static")
app.secret_key = trading_bot.os.getenv("SECRET_KEY", "change-me-in-production-please")
CORS(app)

DASHBOARD_USER     = trading_bot.os.getenv("DASHBOARD_USER", "lennert")
DASHBOARD_PASSWORD = trading_bot.os.getenv("DASHBOARD_PASSWORD", "alphabot2024")

log = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────

_bot_thread  = None
_bot_running = False
_bot_lock    = threading.Lock()
_activity_log = []
_MAX_LOG = 300

def _push_log(msg: str, kind: str = "info"):
    _activity_log.append({"time": datetime.now().strftime("%H:%M:%S"), "message": msg, "type": kind})
    if len(_activity_log) > _MAX_LOG:
        _activity_log.pop(0)

class UIHandler(logging.Handler):
    def emit(self, record):
        kind = "error" if record.levelno >= logging.ERROR else \
               "warn"  if record.levelno >= logging.WARNING else "info"
        _push_log(self.format(record), kind)

ui_handler = UIHandler()
ui_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(ui_handler)

# ── Auth ──────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "unauthorized"}), 401
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated

@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        if (request.form.get("username") == DASHBOARD_USER and
                request.form.get("password") == DASHBOARD_PASSWORD):
            session["logged_in"] = True
            session.permanent = True
            return redirect("/")
        error = "Invalid username or password"
    return send_from_directory("static", "login.html") if not error else \
           send_from_directory("static", "login.html"), 401

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ── Scheduler ─────────────────────────────────────────────────────────────────

def _scheduler_loop():
    global _bot_running
    import schedule as sched
    sched.clear()

    sched.every(30).minutes.do(trading_bot.scan_and_trade)

    for day in ["monday","tuesday","wednesday","thursday","friday"]:
        getattr(sched.every(), day).at("20:45").do(trading_bot.end_of_day_run)

    sched.every().hour.do(trading_bot.print_portfolio_summary)

    _push_log("24/7 scheduler started — crypto always on, stocks during NYSE hours", "info")

    while _bot_running:
        sched.run_pending()
        time.sleep(30)

    sched.clear()
    _push_log("Scheduler stopped", "warn")

# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/account")
@login_required
def api_account():
    try:
        acc = trading_bot.get_account()
        return jsonify({
            "equity":          float(acc.equity),
            "cash":            float(acc.cash),
            "buying_power":    float(acc.buying_power),
            "portfolio_value": float(acc.portfolio_value),
            "status":          str(acc.status),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/positions")
@login_required
def api_positions():
    try:
        result = []
        for sym, p in trading_bot.get_positions().items():
            result.append({
                "symbol":          sym,
                "is_crypto":       trading_bot.is_crypto(sym),
                "qty":             float(p.qty),
                "avg_entry":       float(p.avg_entry_price),
                "current_price":   float(p.current_price),
                "market_value":    float(p.market_value),
                "unrealized_pl":   float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/orders")
@login_required
def api_orders():
    try:
        result = []
        for o in trading_bot.get_open_orders():
            result.append({
                "id":     str(o.id),
                "symbol": o.symbol,
                "side":   str(o.side),
                "qty":    str(o.qty),
                "status": str(o.status),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan")
@login_required
def api_scan():
    try:
        results = []
        market_open = trading_bot.is_market_open()
        for sym in trading_bot.UNIVERSE:
            tradeable = trading_bot.is_crypto(sym) or market_open
            score, signals = trading_bot.score_symbol(sym)
            results.append({
                "symbol":    sym,
                "is_crypto": trading_bot.is_crypto(sym),
                "tradeable": tradeable,
                "score":     score,
                "signals":   signals,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/bot/status")
@login_required
def api_bot_status():
    return jsonify({
        "running":      _bot_running,
        "market_open":  trading_bot.is_market_open(),
        "positions":    len(trading_bot.get_positions()),
    })

@app.route("/api/bot/start", methods=["POST"])
@login_required
def api_bot_start():
    global _bot_thread, _bot_running
    with _bot_lock:
        if _bot_running:
            return jsonify({"status": "already running"})
        _bot_running = True
        _bot_thread  = threading.Thread(target=_scheduler_loop, daemon=True)
        _bot_thread.start()
    return jsonify({"status": "started"})

@app.route("/api/bot/stop", methods=["POST"])
@login_required
def api_bot_stop():
    global _bot_running
    with _bot_lock:
        _bot_running = False
    return jsonify({"status": "stopped"})

@app.route("/api/bot/scan", methods=["POST"])
@login_required
def api_bot_scan():
    threading.Thread(target=trading_bot.scan_and_trade, daemon=True).start()
    return jsonify({"status": "scan triggered"})

@app.route("/api/bot/eod", methods=["POST"])
@login_required
def api_bot_eod():
    threading.Thread(target=trading_bot.end_of_day_run, daemon=True).start()
    return jsonify({"status": "eod triggered"})

@app.route("/api/bot/close_all", methods=["POST"])
@login_required
def api_bot_close_all():
    trading_bot.cancel_all_orders()
    trading_bot.close_all_positions()
    return jsonify({"status": "all closed"})

@app.route("/api/log")
@login_required
def api_log():
    return jsonify(list(reversed(_activity_log)))

@app.route("/")
@login_required
def index():
    return send_from_directory("static", "index.html")

@app.route("/login")
def login_page():
    if session.get("logged_in"):
        return redirect("/")
    return send_from_directory("static", "login.html")

if __name__ == "__main__":
    _push_log("AlphaBot dashboard started", "info")
    app.run(host="0.0.0.0", port=5050, debug=False)
