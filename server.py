"""
Flask server — AlphaBot dashboard + settings API.
"""

import threading
import time
import logging
from datetime import datetime
from functools import wraps
from flask import Flask, jsonify, send_from_directory, request, session, redirect
from flask_cors import CORS
import bot as trading_bot

app = Flask(__name__, static_folder="static")
app.secret_key = trading_bot.os.getenv("SECRET_KEY", "change-me-in-prod")
CORS(app)

log = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────

_bot_thread  = None
_bot_running = False
_bot_lock    = threading.Lock()
_activity_log = []

def _push_log(msg: str, kind: str = "info"):
    _activity_log.append({"time": datetime.now().strftime("%H:%M:%S"), "message": msg, "type": kind})
    if len(_activity_log) > 300:
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
    if request.method == "GET":
        if session.get("logged_in"):
            return redirect("/")
        return send_from_directory("static", "login.html")

    s = trading_bot.load_settings()
    if (request.form.get("username") == s["dashboard_user"] and
            request.form.get("password") == s["dashboard_password"]):
        session["logged_in"] = True
        session.permanent = True
        return redirect("/")
    return send_from_directory("static", "login.html"), 401

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ── Scheduler ─────────────────────────────────────────────────────────────────

def _scheduler_loop():
    global _bot_running
    import schedule as sched
    sched.clear()

    def _reschedule():
        """Rebuild schedule from current settings."""
        sched.clear()
        interval = trading_bot.load_settings().get("scan_interval_min", 30)
        sched.every(interval).minutes.do(trading_bot.scan_and_trade)

        eod_time = trading_bot.load_settings().get("eod_time", "20:45")
        for day in ["monday","tuesday","wednesday","thursday","friday"]:
            getattr(sched.every(), day).at(eod_time).do(trading_bot.end_of_day_run)

        sched.every().hour.do(trading_bot.print_portfolio_summary)
        sched.every(60).minutes.do(_reschedule)  # reload settings every hour
        _push_log(f"Scheduler active — scanning every {interval} min | EOD at {eod_time}", "info")

    _reschedule()

    while _bot_running:
        sched.run_pending()
        time.sleep(20)

    sched.clear()
    _push_log("Bot stopped", "warn")

# ── API: account / positions / orders ─────────────────────────────────────────

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
                "id": str(o.id), "symbol": o.symbol,
                "side": str(o.side), "qty": str(o.qty), "status": str(o.status),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan")
@login_required
def api_scan():
    try:
        s = trading_bot.load_settings()
        market_open = trading_bot.is_market_open()
        results = []
        for sym in trading_bot.get_universe():
            score, signals = trading_bot.score_symbol(sym)
            results.append({
                "symbol":    sym,
                "is_crypto": trading_bot.is_crypto(sym),
                "tradeable": trading_bot.is_crypto(sym) or market_open,
                "score":     score,
                "signals":   signals,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── API: bot control ──────────────────────────────────────────────────────────

@app.route("/api/bot/status")
@login_required
def api_bot_status():
    s = trading_bot.load_settings()
    return jsonify({
        "running":     _bot_running,
        "market_open": trading_bot.is_market_open(),
        "positions":   len(trading_bot.get_positions()),
        "scan_interval": s.get("scan_interval_min", 30),
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

# ── API: settings ─────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["GET"])
@login_required
def api_settings_get():
    return jsonify(trading_bot.load_settings())

@app.route("/api/settings", methods=["POST"])
@login_required
def api_settings_post():
    try:
        current = trading_bot.load_settings()
        updates = request.json

        # Type coercion & validation
        num_fields = ["max_capital","position_size","take_profit_pct",
                      "stop_loss_pct","scan_interval_min","long_threshold","short_threshold"]
        bool_fields = ["enable_stocks","enable_crypto","enable_memecoins",
                       "enable_shorts","eod_close_stocks"]
        str_fields  = ["eod_time","dashboard_user","dashboard_password"]
        list_fields = ["stock_universe","crypto_universe"]

        for k in num_fields:
            if k in updates:
                current[k] = float(updates[k]) if "pct" in k else \
                              int(updates[k])   if k in ["scan_interval_min","long_threshold","short_threshold"] else \
                              float(updates[k])
        for k in bool_fields:
            if k in updates:
                current[k] = bool(updates[k])
        for k in str_fields:
            if k in updates and updates[k]:
                current[k] = str(updates[k])
        for k in list_fields:
            if k in updates:
                current[k] = [s.strip().upper() for s in updates[k] if s.strip()]

        trading_bot.save_settings(current)
        _push_log("⚙️ Settings updated", "warn")
        return jsonify({"status": "saved", "settings": current})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    return send_from_directory("static", "index.html")

@app.route("/settings")
@login_required
def settings_page():
    return send_from_directory("static", "settings.html")

if __name__ == "__main__":
    _push_log("AlphaBot started", "info")
    port = int(trading_bot.os.getenv("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
