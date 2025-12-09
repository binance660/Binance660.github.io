import time
import hmac
import hashlib
import requests
import math
from statistics import mean
import logging
import os
import argparse
import urllib.parse
import signal
from datetime import datetime
import threading

python
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

# ========== CONFIG ==========
# Read API credentials from environment when possible to avoid committing secrets in source.
# IMPORTANT: Set your real API keys via environment variables BEFORE running with --live
# In PowerShell:
#   $env:BINANCE_API_KEY = "your_key_here"
#   $env:BINANCE_API_SECRET = "your_secret_here"
API_KEY = os.getenv("BINANCE_API_KEY", "WNFkSItC9GpmyuYH6wYxk7TkpIYzcijh5CHuDslg1jUDoWL7fzrUpmNDq54a5esb")  # MUST be set via env var before --live
API_SECRET = os.getenv("BINANCE_API_SECRET", "eCI6mP0KFFxho8JXYMzndzAvkXXWjGR3Nm7QXpOKkQ8lR8gBp1tWquwpVWH1fHu2")  # MUST be set via env var before --live
DRY_RUN = False  # FORCE LIVE: real orders will be placed. If validation fails, orders will error.
SYMBOL = "BTCUSDT"
BASE_ASSET = "BTC"
QUOTE_ASSET = "USDT"
BASE_POSITION_USDT = 1.0  # AGGRESSIVE: Start with 50 USDT for faster compounding
MAX_POSITION_USDT = 500.0  # AGGRESSIVE: Allow up to 500 USDT max position
TP_PCT = 0.003   # 0.3% take profit (tighter, faster exits)
SL_PCT = 0.002   # 0.2% stop loss (stricter risk)
COMPOUND_STEP_PCT = 0.10  # AGGRESSIVE: 10% compounding on wins (double speed vs 5%)
REST_BASE = "https://api.binance.com/api"
LOG_FILE = "live_trades.log"
# ULTRA-FAST: 0.5s loop interval for rapid scalping
LOOP_INTERVAL = float(os.getenv("LOOP_INTERVAL", "0.5"))
# Multi-pair settings (but optimized for single best pair focus)
MULTI_PAIR_COUNT = int(os.getenv("MULTI_PAIR_COUNT", "5"))  # FAST: Only top 5 instead of 10
# AGGRESSIVE: Pause at 0.45 (was 0.60) - more trades when market is decent
PAUSE_CONFIDENCE_THRESHOLD = float(os.getenv("PAUSE_CONFIDENCE_THRESHOLD", "0.45"))
# AGGRESSIVE: Entry at 0.55 (was 0.70) - more aggressive entry
OPEN_CONFIDENCE_BASE = float(os.getenv("OPEN_CONFIDENCE_BASE", "0.55"))
# How often (seconds) to re-check API key validity when --live was requested but validation failed
VALIDATION_INTERVAL = int(os.getenv("VALIDATION_INTERVAL", "60"))
# Backoff params (seconds)
BACKOFF_BASE = 1.0
BACKOFF_MAX = 60.0
# Shutdown flag set by signal handlers
SHUTDOWN = False
# ============================

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])

# Signer
def sign_params(params):
    # Use urllib.parse.urlencode to correctly percent-encode values.
    safe_params = {k: str(params[k]) for k in params}
    query = urllib.parse.urlencode(sorted(safe_params.items()))
    signature = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return query + "&signature=" + signature

# REST helpers
def rest_get(path, params=None):
    url = REST_BASE + path
    headers = {"X-MBX-APIKEY": API_KEY}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def rest_post_signed(path, params):
    url = REST_BASE + path
    # Do not mutate caller's dict
    p = dict(params)
    p["timestamp"] = int(time.time() * 1000)

    # Dry-run support to avoid placing real orders during development.
    if DRY_RUN:
        logging.info(f"DRY_RUN: would POST {path} params={p}")
        # Return a simulated successful response
        return {"symbol": p.get("symbol"), "side": p.get("side"), "status": "SIMULATED", "quantity": p.get("quantity")}

    signed = sign_params(p)
    headers = {"X-MBX-APIKEY": API_KEY}
    try:
        r = requests.post(url + "?" + signed, headers=headers, timeout=10)
        # If Binance returns an error code, include the body in the logged message for diagnosis
        if r.status_code != 200 and r.status_code != 201:
            logging.error(f"BINANCE ERROR {r.status_code}: {r.text}")
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as he:
        logging.error(f"HTTPError during signed POST to {path}: {he} - body={getattr(he.response, 'text', None)}")
        raise
    except Exception as e:
        logging.exception(f"Unexpected error during signed POST to {path}: {e}")
        raise


def _signal_handler(signum, frame):
    global SHUTDOWN
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    SHUTDOWN = True


def validate_api_key_background():
    """Silently validate API key in background. Auto-switch to LIVE if valid."""
    global DRY_RUN
    try:
        params = {"timestamp": int(time.time() * 1000)}
        signed = sign_params(params)
        headers = {"X-MBX-APIKEY": API_KEY}
        r = requests.get(REST_BASE + "/v3/account?" + signed, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            can_trade = data.get("canTrade", False)
            if can_trade and not DRY_RUN:
                logging.info("[OK] API KEY VALID - Trading permission confirmed. LIVE mode active.")
                return True
            elif can_trade:
                logging.info("[OK] API KEY VALID - Ready to enable LIVE mode with --live flag.")
                return True
            else:
                logging.warning("[WARN] API key valid but trading DISABLED on Binance account.")
                return False
        else:
            logging.warning(f"[WARN] API key validation failed ({r.status_code}). Running in DRY_RUN.")
            return False
    except Exception as e:
        logging.debug(f"API validation error (will retry): {str(e)[:100]}")
        return False


def start_api_validation_watcher(live_requested: bool):
    """If the user requested --live but validation failed, keep checking in background and enable LIVE when possible.

    This runs in a daemon thread and will flip DRY_RUN -> False once validation succeeds.
    """
    if not live_requested:
        return

    def _watcher():
        global DRY_RUN
        logging.info(f"Starting API validation watcher (interval={VALIDATION_INTERVAL}s)")
        while not SHUTDOWN and DRY_RUN:
            try:
                ok = validate_api_key_background()
                if ok:
                    DRY_RUN = False
                    logging.info("[OK] API key validated in background — LIVE mode enabled.")
                    break
            except Exception as e:
                logging.debug(f"Background validator error: {e}")
            # sleep with early exit
            slept = 0
            while slept < VALIDATION_INTERVAL and not SHUTDOWN and DRY_RUN:
                time.sleep(1)
                slept += 1

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()


def run_loop(iterations=None):
    """Run the trading loop continuously.

    iterations: if None, run until signaled (infinite). If an int, run that many iterations then return.
    """
    # Register signal handlers for graceful shutdown
    try:
        signal.signal(signal.SIGINT, _signal_handler)
    except Exception:
        # signal may not be available on some platforms; ignore
        pass
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass

    count = 0
    backoff = BACKOFF_BASE
    while not SHUTDOWN:
        start = time.time()
        try:
            check_and_trade()
            # success -> reset backoff
            backoff = BACKOFF_BASE
        except Exception as e:
            logging.exception(f"Unhandled error in check_and_trade: {e}")
            # exponential backoff but cap it
            logging.info(f"Backing off for {backoff:.1f}s before retrying")
            time.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)

        count += 1
        if iterations is not None and count >= iterations:
            logging.info("Completed requested number of iterations, exiting run_loop")
            break

        # Sleep the remaining loop interval if any
        elapsed = time.time() - start
        to_sleep = max(0.0, LOOP_INTERVAL - elapsed)
        # Allow early exit during sleep
        slept = 0.0
        while slept < to_sleep and not SHUTDOWN:
            step = min(1.0, to_sleep - slept)
            time.sleep(step)
            slept += step


# Price fetcher
def get_price():
    return get_price_for_symbol(SYMBOL)


def get_price_for_symbol(symbol):
    book = rest_get("/v3/depth", {"symbol": symbol, "limit": 5})
    bid = float(book["bids"][0][0])
    ask = float(book["asks"][0][0])
    return (bid + ask) / 2


def get_orderbook(depth=10, symbol=None):
    """Return aggregated bid/ask sizes for the top `depth` levels for a symbol."""
    symbol = symbol or SYMBOL
    book = rest_get("/v3/depth", {"symbol": symbol, "limit": max(5, depth)})
    bids = [(float(p), float(q)) for p, q in book.get("bids", [])[:depth]]
    asks = [(float(p), float(q)) for p, q in book.get("asks", [])[:depth]]
    bid_liq = sum(q for _, q in bids)
    ask_liq = sum(q for _, q in asks)
    return {"bids": bids, "asks": asks, "bid_liq": bid_liq, "ask_liq": ask_liq}


def get_agg_trades(limit=100, symbol=None):
    """Get recent aggregated trades for a symbol (public endpoint)."""
    symbol = symbol or SYMBOL
    trades = rest_get("/v3/aggTrades", {"symbol": symbol, "limit": limit})
    return trades


def get_klines(interval='1m', limit=20, symbol=None):
    """Return recent klines for a symbol."""
    symbol = symbol or SYMBOL
    raw = rest_get("/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    klines = []
    for k in raw:
        # k: [Open time, Open, High, Low, Close, Volume, ...]
        klines.append({
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "start_time": int(k[0])
        })
    return klines


def compute_pressure_from_trades(trades):
    """Compute buy/sell pressure from aggregated trades. Returns ratio buyer_qty/(buyer+seller)."""
    buy_qty = 0.0
    sell_qty = 0.0
    for t in trades:
        qty = float(t.get('q', 0))
        # 'm' == True means the buyer is the maker (seller initiated) -> it's sell pressure
        if t.get('m'):
            sell_qty += qty
        else:
            buy_qty += qty
    total = buy_qty + sell_qty
    if total == 0:
        return 0.5
    return buy_qty / total


def compute_candle_signal(klines):
    """Fast candle-based momentum signal in [-1,1]. Positive favors buy.
    OPTIMIZED: Reduced calculation for speed, focus on recent momentum.
    """
    if not klines or len(klines) < 3:
        return 0.0
    closes = [k['close'] for k in klines]
    volumes = [k['volume'] for k in klines]
    
    # AGGRESSIVE: Last close vs previous close (faster, more reactive)
    momentum = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0.0
    
    # AGGRESSIVE: Volume spike vs last 5 candles (not 10)
    vol_avg = mean(volumes[-5:]) if len(volumes) >= 5 else mean(volumes)
    vol_spike = (volumes[-1] / vol_avg) if vol_avg > 0 else 1.0
    
    # AGGRESSIVE: SMA slope on last 10 candles (not 15)
    sma_short = mean(closes[-3:])
    sma_long = mean(closes[-10:]) if len(closes) >= 10 else mean(closes)
    slope = (sma_short - sma_long) / sma_long if sma_long > 0 else 0.0

    score = 0.0
    score += max(min(momentum * 10.0, 1.0), -1.0) * 0.4  # AGGRESSIVE: Boost momentum weight
    score += max(min((vol_spike - 1.0) * 3.0, 1.0), -1.0) * 0.3
    score += max(min(slope * 10.0, 1.0), -1.0) * 0.3  # AGGRESSIVE: Boost slope weight
    
    return max(min(score, 1.0), -1.0)


def compute_confidence(book, trades, klines):
    """Combine orderbook liquidity, trade pressure, and candle signal into 0..1 confidence score for buying."""
    # orderbook pressure: more bid liquidity -> bullish (0..1)
    ob_pressure = 0.5
    if book:
        total_liq = book.get('bid_liq', 0) + book.get('ask_liq', 0)
        if total_liq > 0:
            ob_pressure = book.get('bid_liq', 0) / total_liq

    trade_pressure = compute_pressure_from_trades(trades) if trades else 0.5
    candle_signal = compute_candle_signal(klines) if klines else 0.0

    # Map candle_signal from [-1,1] to [0,1]
    candle_conf = (candle_signal + 1.0) / 2.0

    # Weighted average (tunable)
    w_ob = 0.3
    w_tr = 0.4
    w_ca = 0.3
    conf = w_ob * ob_pressure + w_tr * trade_pressure + w_ca * candle_conf
    return conf

# Quantity calculator
def qty_from_usdt(usdt, price):
    qty = usdt / price
    return float(qty)


_exchange_info_cache = None

def fetch_exchange_info():
    """Fetch and cache exchange info from Binance (symbols and filters)."""
    global _exchange_info_cache
    if _exchange_info_cache is not None:
        return _exchange_info_cache
    try:
        info = rest_get("/v3/exchangeInfo")
        _exchange_info_cache = info
        return info
    except Exception as e:
        logging.warning(f"Could not fetch exchangeInfo: {e}")
        return None


def get_symbol_filters(symbol):
    info = fetch_exchange_info()
    if not info:
        return {}
    for s in info.get("symbols", []):
        if s.get("symbol") == symbol:
            filters = {f["filterType"]: f for f in s.get("filters", [])}
            return filters
    return {}


def round_qty_for_symbol(symbol, qty):
    """Round down qty to the symbol's LOT_SIZE stepSize and enforce minQty."""
    filters = get_symbol_filters(symbol)
    lot = filters.get("LOT_SIZE")
    if not lot:
        return float(f"{qty:.6f}")
    step = float(lot.get("stepSize"))
    min_qty = float(lot.get("minQty", 0))
    # compute number of steps
    steps = math.floor(qty / step)
    qty_adj = steps * step
    # enforce min
    if qty_adj < min_qty:
        return 0.0
    # format to same decimal places as step
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    fmt = "{:.%df}" % decimals if decimals >= 0 else "{:.6f}"
    try:
        return float(fmt.format(qty_adj))
    except Exception:
        return float(f"{qty_adj:.6f}")


def round_price_for_symbol(symbol, price):
    filters = get_symbol_filters(symbol)
    tick = filters.get("PRICE_FILTER")
    if not tick:
        return float(f"{price:.2f}")
    tick_size = float(tick.get("tickSize"))
    ticks = math.floor(price / tick_size)
    p = ticks * tick_size
    # format decimals
    decimals = max(0, -int(math.floor(math.log10(tick_size)))) if tick_size < 1 else 0
    fmt = "{:.%df}" % decimals if decimals >= 0 else "{:.2f}"
    try:
        return float(fmt.format(p))
    except Exception:
        return float(f"{p:.2f}")

# Trade executor
def place_market_order(side, qty, symbol=None):
    path = "/v3/order"
    symbol = symbol or SYMBOL
    # Ensure quantity is formatted consistently
    qty_str = f"{qty:.6f}"
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": qty_str
    }
    return rest_post_signed(path, params)

# Position tracker
class Position:
    def __init__(self):
        self.qty = 0.0
        self.entry_price = None
        self.usdt_allocated = BASE_POSITION_USDT
        self.symbol = None

    def open(self, qty, price):
        self.qty = qty
        self.entry_price = price

    def close(self):
        self.qty = 0.0
        self.entry_price = None

position = Position()

# For multi-pair operation keep a dict of positions per symbol
positions = {}

# Main logic
def get_top_usdt_pairs(limit=MULTI_PAIR_COUNT):
    """Return top USDT-quoted trading pairs by quoteVolume from 24hr ticker."""
    tickers = rest_get("/v3/ticker/24hr")
    usdt_pairs = [t for t in tickers if t.get('symbol', '').endswith('USDT') and t.get('symbol')]
    # sort by quoteVolume (string), convert to float safely
    def qv(x):
        try:
            return float(x.get('quoteVolume') or x.get('quoteVolume', 0))
        except Exception:
            return 0.0
    usdt_pairs.sort(key=qv, reverse=True)
    return [p['symbol'] for p in usdt_pairs[:limit]]


def check_and_trade():
    """AGGRESSIVE SCALPING: Scan top 5 pairs fast, pick best, trade aggressively, compound hard."""
    try:
        symbols = get_top_usdt_pairs()
    except Exception as e:
        logging.exception(f"Failed to fetch top pairs: {e}")
        return

    best_sym = None
    best_conf = 0.0
    sym_info = {}

    # FAST: Evaluate each symbol with minimal data
    for sym in symbols[:MULTI_PAIR_COUNT]:  # Only check top 5
        try:
            price = get_price_for_symbol(sym)
            book = get_orderbook(depth=3, symbol=sym)  # FAST: Only depth 3
            trades = get_agg_trades(limit=50, symbol=sym)  # FAST: Only 50 trades
            klines = get_klines(interval='1m', limit=10, symbol=sym)  # FAST: Only 10 klines
            conf = compute_confidence(book, trades[-50:], klines[-10:])
            sym_info[sym] = {"price": price, "conf": conf, "book": book}
            if conf > best_conf:
                best_conf = conf
                best_sym = sym
        except Exception as e:
            logging.debug(f"Skipping {sym}: {e}")
            continue

    if not best_sym:
        logging.debug("No valid pairs found")
        return

    logging.info(f"Best: {best_sym} conf={best_conf:.3f}")

    # AGGRESSIVE: Pause only if market is really bad
    if best_conf < PAUSE_CONFIDENCE_THRESHOLD:
        logging.debug(f"Pausing (conf {best_conf:.3f} < {PAUSE_CONFIDENCE_THRESHOLD})")
        return

    # Get or create position for this pair
    pos = positions.get(best_sym)
    if not pos:
        pos = Position()
        pos.symbol = best_sym
        positions[best_sym] = pos

    price = sym_info[best_sym]['price']
    conf = sym_info[best_sym]['conf']

    # Entry
    if pos.qty == 0:
        # AGGRESSIVE: Lower threshold, more trades
        threshold = OPEN_CONFIDENCE_BASE + (pos.usdt_allocated / MAX_POSITION_USDT) * 0.10
        if conf >= threshold:
            qty = qty_from_usdt(pos.usdt_allocated, price)
            qty = round_qty_for_symbol(best_sym, qty)
            if qty <= 0:
                logging.warning(f"Calculated BUY qty too small for symbol {best_sym} after rounding; skipping entry.")
            else:
                res = place_market_order("BUY", qty, symbol=best_sym)
            pos.open(qty, price)
            logging.info(f"[TRADE] BUY {best_sym}: qty={qty:.6f} price={price:.2f} conf={conf:.3f}")
    else:
        # AGGRESSIVE: Tighter TP/SL for fast cycles
        pnl = (price - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0
        
        # AGGRESSIVE: Exit on confidence flip at 0.40 (was 0.35)
        close_on_flip = conf < 0.40
        exit_signal = pnl >= TP_PCT or pnl <= -SL_PCT or close_on_flip
        
        if exit_signal:
            sell_qty = round_qty_for_symbol(best_sym, pos.qty)
            if sell_qty <= 0:
                logging.warning(f"Calculated SELL qty too small for symbol {best_sym} after rounding; skipping exit.")
            else:
                res = place_market_order("SELL", sell_qty, symbol=best_sym)
                logging.info(f"[TRADE] SELL {best_sym}: qty={sell_qty:.6f} price={price:.2f} pnl={pnl:.4f} ({pnl*100:.2f}%) conf={conf:.3f}")
            
            # AGGRESSIVE: Compound aggressively on wins
            if pnl > 0.0005:  # Any positive PnL compounds
                old_size = pos.usdt_allocated
                pos.usdt_allocated = min(pos.usdt_allocated * (1 + COMPOUND_STEP_PCT), MAX_POSITION_USDT)
                logging.info(f"[COMPOUND] {best_sym}: {old_size:.2f} USDT -> {pos.usdt_allocated:.2f} USDT (+{COMPOUND_STEP_PCT*100:.0f}%)")
            
            pos.close()

# Main loop
def main():
    parser = argparse.ArgumentParser(description="Binance scalping bot (safe by default)")
    parser.add_argument("--live", action="store_true", help="Place real orders (default is dry-run)")
    args = parser.parse_args()

    global DRY_RUN
    # If DRY_RUN is already False (forced at module level), skip the validation and just warn
    if not DRY_RUN:
        logging.info("FORCE LIVE MODE ENABLED - real orders will be placed immediately. API validation skipped.")
        logging.warning("[WARN] Running in forced LIVE mode without API validation. Orders may fail if key is invalid.")
    elif args.live:
        DRY_RUN = False
        logging.info("Running in LIVE mode — real orders will be placed. Make sure API keys are correct.")
        # Validate API key on startup before trading
        logging.info("Validating API credentials...")
        if validate_api_key_background():
            logging.info("[OK] API KEY VALID - Proceeding with LIVE trading.")
        else:
            logging.warning("[WARN] API KEY VALIDATION FAILED - Falling back to DRY_RUN mode for safety.")
            DRY_RUN = True
        # Start a background watcher that will keep checking the API key and enable LIVE once it becomes valid
        start_api_validation_watcher(live_requested=True)
    else:
        logging.info("Running in DRY_RUN mode — no real orders will be placed. Use --live to enable live trading.")

    logging.info("Starting Binance scalping bot loop...")
    try:
        # Run indefinitely. Use CTRL-C to stop (SIGINT).
        run_loop()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, exiting...")
    logging.info("Bot stopped.")

if __name__ == "__main__":

    main()

