# combined_nifty_atm0909_vwap.py
# NIFTY ΔOI Imbalance + TradingView VWAP alert
# - ATM: TV 09:09 → Yahoo daily open (robust, no-verify) → NSE underlying (provisional)
# - TV loop immediately upgrades ATM when 09:09 appears
# - Manual ATM override in sidebar
# - Yahoo 429 tolerant, retries, query1/query2, verify=False
# - OC loop reloads ATM store every cycle
# - Weekday neighbors: Fri/Sat/Sun ±5, Mon ±4, Tue ±3, Wed ±2, Thu ±1
# - VWAP 15m session from TV 1m candles
# - Full logging + CSV/text outputs

import os, json, time, base64, datetime as dt, pathlib, threading, warnings, logging, sys, math, random
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import yfinance as yf
import certifi
import requests, urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= USER SETTINGS =================
SYMBOL               = "NIFTY"
FETCH_EVERY_SECONDS  = 60          # option-chain poll (1 min)
TV_FETCH_SECONDS     = 60           # TradingView poll (1 min)
AUTOREFRESH_MS       = 10_000

OUT_DIR              = pathlib.Path.home() / "Documents" / "NSE_output"
CSV_PATH             = OUT_DIR / "nifty_currweek_change_oi_atm_dynamic.csv"
ATM_STORE_PATH       = OUT_DIR / "nifty_atm_store.json"
LOG_PATH             = OUT_DIR / "nifty_app.log"
VWAP_NOW_TXT         = OUT_DIR / "nifty_vwap_now.txt"
VWAP_LOG_CSV         = OUT_DIR / "nifty_vwap_log.csv"

MAX_NEIGHBORS_LIMIT  = 20
IMBALANCE_TRIGGER    = 20.0         # %
VWAP_TOLERANCE_PTS   = 5.0          # alert when |spot - vwap| <= tolerance

# ---- HARD-CODED TradingView credentials (REPLACE THESE) ----
TV_USERNAME          = "YOUR_TV_USERNAME"
TV_PASSWORD          = "YOUR_TV_PASSWORD"
# ============================================================

API_URL  = "https://www.nseindia.com/api/option-chain-indices"
IST      = dt.timezone(dt.timedelta(hours=5, minutes=30))

BASE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"https://www.nseindia.com/option-chain?symbol={SYMBOL}",
}

# ensure certifi is used by libs that honor SSL_CERT_FILE
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# --- tiny 0.1s beep WAV (base64) ---
BEEP_WAV_B64 = (
    "UklGRmYAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABYAAABhY2NkZGdn"
    "aGhoaWlpamptbW1tbm5ub29wcHBxcXFycnJzc3N0dHR1dXV2dnZ3d3d4eHh5"
    "eXl6enp7e3t8fHx9fX1+fn5/f3+AgICAgoKCg4ODhISEhYWFhoaGiIiIkJCQ"
    "kZGRkpKSlJSUlZWVmZmZmpqamsrKy8vLzMzMzc3Nzs7O0NDQ0dHR0lJSU1NT"
    "U9PT1NTU1dXV1paWmZmZmpqam5ubnBwcHJycnR0dHZ2dnd3d3h4eXl5enp6f"
    "Hx8fX19fn5+f39/gICA"
)

# ---------------- Logging ----------------
def setup_logger():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nifty_app")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt); ch.setLevel(logging.INFO)

    logger.addHandler(fh); logger.addHandler(ch)
    logger.info("Logger initialized. Log file: %s", LOG_PATH)
    return logger

log = setup_logger()

def now_ist() -> dt.datetime:
    return dt.datetime.now(IST)

def today_str() -> str:
    return now_ist().strftime("%Y%m%d")

def load_atm_store() -> dict:
    if ATM_STORE_PATH.exists():
        try:
            return json.loads(ATM_STORE_PATH.read_text())
        except Exception as e:
            log.error("Failed to read ATM store: %s", e)
    return {}

def save_atm_store(store: dict):
    try:
        ATM_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        ATM_STORE_PATH.write_text(json.dumps(store, indent=2))
    except Exception as e:
        log.error("Failed to write ATM store: %s", e)

def update_store_atm(atm: int, base_value: float, status: str):
    """Atomic update of the ATM store (used by both loops)."""
    store = load_atm_store()
    store.update({
        "date": today_str(),
        "atm_strike": int(atm),
        "base_value": float(base_value),
        "atm_status": status
    })
    save_atm_store(store)
    log.info("Store ATM updated -> %s (%s, base=%.2f)", atm, status, base_value)

# ---------------- NSE OPTION-CHAIN ----------------
def new_session():
    try:
        import cloudscraper
        warnings.filterwarnings("ignore", category=UserWarning, module="cloudscraper")
        s = cloudscraper.create_scraper(delay=8, browser={'browser':'chrome','platform':'windows'})
        log.info("Created cloudscraper session")
    except ModuleNotFoundError:
        import requests as _rq
        s = _rq.Session()
        log.warning("cloudscraper not installed; using requests.Session")
    s.headers.update(BASE_HEADERS)
    try:
        s.get(f"https://www.nseindia.com/option-chain?symbol={SYMBOL}", timeout=8)
    except Exception as e:
        log.warning("Handshake to NSE failed (continuing): %s", e)
    return s

def pick_current_week_expiry(raw: dict) -> str | None:
    today = now_ist().date()
    parsed = []
    for s in raw.get("records", {}).get("expiryDates", []):
        try:
            parsed.append((s, dt.datetime.strptime(s, "%d-%b-%Y").date()))
        except Exception:
            pass
    if not parsed:
        log.error("No expiryDates in JSON.")
        return None
    future = [p for p in parsed if p[1] >= today]
    chosen = min(future, key=lambda x: x[1]) if future else min(parsed, key=lambda x: x[1])
    return chosen[0]

def round_to_50(x: float) -> int:
    return int(round(x / 50.0) * 50)

def fetch_raw_option_chain():
    s = new_session()
    for i in range(6):
        try:
            r = s.get(API_URL, params={"symbol": SYMBOL}, timeout=10)
            if r.status_code == 200:
                try:
                    raw = r.json()
                    if "records" in raw and "data" in raw["records"]:
                        log.info("OC fetch OK on attempt %d", i+1)
                        return raw
                    else:
                        log.warning("OC JSON missing keys on attempt %d", i+1)
                except json.JSONDecodeError as e:
                    log.warning("OC JSON decode error on attempt %d: %s", i+1, e)
            else:
                log.warning("OC HTTP %s on attempt %d", r.status_code, i+1)
        except Exception as e:
            log.warning("OC fetch exception on attempt %d: %s", i+1, e)
        time.sleep(2)
    log.error("OC fetch failed after retries.")
    return None

# ---------------- TradingView helpers ----------------
def tv_login():
    from tvDatafeed import TvDatafeed
    try:
        tv = TvDatafeed(username="dileep.marchetty@gmail.com", password="1dE6Land@123")
        log.info("Logged in to TradingView as %s", TV_USERNAME)
        return tv
    except Exception as e:
        log.error("TradingView login failed: %s", e)
        raise

def fetch_tv_1m_session():
    """Fetch latest 1m NIFTY candles from TV, tz-aware IST; retry a few times."""
    try:
        from tvDatafeed import Interval
    except Exception as e:
        log.error("tvDatafeed import failed: %s", e)
        return None

    last_err = None
    for i in range(1, 5):
        try:
            tv = tv_login()
            df = tv.get_hist(symbol="NIFTY", exchange="NSE",
                             interval=Interval.in_1_minute, n_bars=2000)
            if df is not None and not df.empty:
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
                else:
                    df.index = df.index.tz_convert("Asia/Kolkata")
                log.info("TV 1m bars fetched on attempt %d. Last: %s", i, df.index.max())
                return df
            last_err = "empty dataframe"
            log.warning("TV fetch attempt %d: empty dataframe", i)
        except Exception as e:
            last_err = e
            log.warning("TV fetch attempt %d failed: %s", i, e)
        time.sleep(2 + (i-1)*2)
    log.error("TV fetch 1m failed after retries: %s", last_err)
    return None

def price_at_0909(df_1m: pd.DataFrame) -> float | None:
    """Close at 09:09 IST of latest session; fallback nearest 09:05–09:14; else 09:15 open."""
    if df_1m is None or df_1m.empty:
        return None
    latest_date = df_1m.index.max().date()
    t909 = dt.datetime.combine(latest_date, dt.time(9, 9), tzinfo=IST)
    try:
        if t909 in df_1m.index:
            return float(df_1m.loc[t909, "close"])
        win = df_1m.between_time("09:05", "09:14")
        if not win.empty and win.index.date.max() == latest_date:
            idx = min(win.index, key=lambda t: abs((t - t909).total_seconds()))
            return float(win.loc[idx, "close"])
        t915 = dt.datetime.combine(latest_date, dt.time(9, 15), tzinfo=IST)
        if t915 in df_1m.index:
            return float(df_1m.loc[t915, "open"])
    except Exception as e:
        log.error("price_at_0909 error: %s", e)
    return None

# -------- Robust Yahoo daily open (no-verify, retry, query1/query2) --------
_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
]

def _yahoo_chart_json(symbol_enc: str) -> float | None:
    """Try query1 then query2; return latest available daily open."""
    params = {"range": "10d", "interval": "1d", "includePrePost": "false", "events": "div,split"}
    headers = {"User-Agent": random.choice(_UAS), "Accept": "application/json"}
    for host in ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"):
        url = f"{host}/v8/finance/chart/{symbol_enc}"
        r = requests.get(url, params=params, headers=headers, timeout=15, verify=False)
        if r.status_code == 429:
            raise requests.HTTPError("429")
        r.raise_for_status()
        j = r.json()
        result = j["chart"]["result"][0]
        ts = result["timestamp"]
        opens = result["indicators"]["quote"][0]["open"]
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Kolkata")
        df = pd.DataFrame({"open": opens}, index=idx)
        ist_today = now_ist().date()
        row = df[df.index.date == ist_today]
        val = float(row["open"].iloc[0]) if not row.empty else float(df["open"].iloc[-1])
        return val
    return None

def yahoo_open_today_ist() -> float | None:
    """Fetch NIFTY50 daily open from Yahoo, tolerant to 429 and SSL issues."""
    symbol_enc = "%5ENSEI"
    retries = 6
    wait = 2.0
    for i in range(1, retries + 1):
        try:
            val = _yahoo_chart_json(symbol_enc)
            if val and val > 0:
                log.info("Yahoo(open, robust) fetched: %.2f (attempt %d)", val, i)
                return val
        except requests.HTTPError as e:
            if "429" in str(e):
                log.warning("Yahoo 429 rate-limited (attempt %d) — backing off %.1fs", i, wait)
                time.sleep(wait)
                wait = min(wait * 2.0, 30.0)
                continue
            log.error("Yahoo HTTP error (attempt %d): %s", i, e)
        except Exception as e:
            log.warning("Yahoo fetch error (attempt %d): %s", i, e)
        time.sleep(wait)
        wait = min(wait * 1.6, 20.0)
    log.error("Yahoo open failed after retries.")
    return None

# -------- VWAP 15m --------
def compute_session_vwap_15m(df_1m: pd.DataFrame) -> tuple[float | None, pd.DataFrame]:
    """Session VWAP on 15m bars, last trading session fallback."""
    if df_1m is None or df_1m.empty or not isinstance(df_1m.index, pd.DatetimeIndex):
        log.error("compute_session_vwap_15m: invalid df_1m")
        return None, pd.DataFrame()

    df = df_1m.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")

    latest_ts = df.index.max()
    if pd.isna(latest_ts):
        log.error("compute_session_vwap_15m: latest_ts is NaN")
        return None, pd.DataFrame()

    session_date = latest_ts.date()
    start = dt.datetime.combine(session_date, dt.time(9, 15), tzinfo=IST)
    end   = dt.datetime.combine(session_date, dt.time(15, 30), tzinfo=IST)

    df = df[(df.index >= start) & (df.index <= end)]
    if df.empty:
        log.error("compute_session_vwap_15m: empty after session filter")
        return None, pd.DataFrame()

    price = df["close"].astype(float)
    vol   = df["volume"].fillna(0).astype(float)

    df["pv"] = price * vol
    df["cum_vol"] = vol.cumsum()
    df["cum_pv"]  = df["pv"].cumsum()
    df["vwap"]    = df["cum_pv"] / df["cum_vol"].replace({0: math.nan})

    df15 = df.resample("15T").agg({
        "open": "first",
        "high": "max",
        "low":  "min",
        "close":"last",
        "volume":"sum",
        "vwap":"last"
    }).dropna(subset=["close"])

    vwap_latest = float(df15["vwap"].iloc[-1]) if not df15.empty else None
    log.info("VWAP computed: %s", f"{vwap_latest:.2f}" if vwap_latest is not None else "None")
    return vwap_latest, df15

# ---------------- Weekday neighbors mapping ----------------
def neighbors_by_weekday(d: dt.date) -> int:
    # Fri/Sat/Sun -> ±5, Mon -> ±4, Tue -> ±3, Wed -> ±2, Thu -> ±1
    wd = d.weekday()  # Mon=0 .. Sun=6
    mapping = {0: 4, 1: 3, 2: 2, 3: 1, 4: 5, 5: 5, 6: 5}
    return mapping.get(wd, 3)

def nearest_strike_block(strikes_sorted: list[int], atm: int, neighbors_each: int) -> list[int]:
    if not strikes_sorted:
        return []
    if atm not in strikes_sorted:
        atm = min(strikes_sorted, key=lambda x: abs(x - atm))
    idx = strikes_sorted.index(atm)
    lo = max(0, idx - neighbors_each)
    hi = min(len(strikes_sorted) - 1, idx + neighbors_each)
    return strikes_sorted[lo:hi+1]

# ---------------- Build OC DF with imbalance + ATM logic ----------------
def build_df_with_imbalance(raw: dict, store: dict):
    # always refresh from disk to pick up TV-loop upgrades / manual override
    store = load_atm_store()

    if not raw:
        return pd.DataFrame(), None

    expiry = pick_current_week_expiry(raw)
    if not expiry:
        return pd.DataFrame(), None

    records = raw["records"]
    rows = [x for x in records["data"] if x.get("expiryDate") == expiry]
    if not rows:
        log.error("No rows for chosen expiry %s", expiry)
        return pd.DataFrame(), None

    df_all = pd.json_normalize(rows)
    if "strikePrice" not in df_all.columns:
        log.error("strikePrice missing")
        return pd.DataFrame(), None
    strikes_all = sorted({int(v) for v in df_all["strikePrice"].dropna().astype(int)})

    underlying = float(records.get("underlyingValue", 0.0))
    today_date  = now_ist().date()
    today_key   = today_str()

    def capture_today_atm_tv_0909():
        df1 = fetch_tv_1m_session()
        px909 = price_at_0909(df1) if df1 is not None else None
        if px909 and px909 > 0:
            base_val = float(px909)
            guess = round_to_50(base_val)
            atm_local = guess if guess in strikes_all else min(strikes_all, key=lambda x: abs(x - base_val))
            log.info("ATM capture(09:09 TV): base=%.2f atm=%s", base_val, atm_local)
            return int(atm_local), base_val, "captured-0909"
        return None, None, "capture-failed"

    def capture_today_atm_yahoo_open():
        yopen = yahoo_open_today_ist()
        if yopen and yopen > 0:
            base_val = float(yopen)
            guess = round_to_50(base_val)
            atm_local = guess if guess in strikes_all else min(strikes_all, key=lambda x: abs(x - base_val))
            log.info("ATM capture(Yahoo open): base=%.2f atm=%s", base_val, atm_local)
            return int(atm_local), base_val, "captured-yahoo-open"
        return None, None, "capture-failed"

    def capture_today_atm_underlying():
        base_val = underlying
        guess = round_to_50(base_val)
        atm_local = guess if guess in strikes_all else min(strikes_all, key=lambda x: abs(x - base_val))
        log.warning("ATM provisional from underlying: base=%.2f atm=%s", base_val, atm_local)
        return int(atm_local), float(base_val), "provisional"

    stored_date   = store.get("date")
    stored_atm    = store.get("atm_strike")
    stored_status = store.get("atm_status", "unknown")

    need_fresh = (stored_date != today_key)

    if need_fresh:
        atm_strike = None; base_val = None; atm_status = "capture-failed"
        for capt in (capture_today_atm_tv_0909, capture_today_atm_yahoo_open, capture_today_atm_underlying):
            a,b,s = capt()
            if a is not None:
                atm_strike, base_val, atm_status = a,b,s
                break
        update_store_atm(atm_strike, base_val, atm_status)
    else:
        atm_strike = int(stored_atm)
        atm_status = stored_status
        base_val   = store.get("base_value", 0.0)

        if atm_status != "captured-0909" and atm_status != "manual-override":
            y_a, y_b, y_s = capture_today_atm_yahoo_open()
            if y_a is not None and atm_strike != y_a:
                log.info("Correcting ATM via Yahoo: %s → %s", atm_strike, y_a)
                atm_strike, base_val, atm_status = y_a, y_b, y_s
                update_store_atm(atm_strike, base_val, atm_status)

        log.info("Using ATM: %s (%s)", atm_strike, atm_status)

    # neighbors by weekday rule
    neighbors_each = neighbors_by_weekday(today_date)
    neighbors_each = min(neighbors_each, MAX_NEIGHBORS_LIMIT)
    wanted = set(nearest_strike_block(strikes_all, atm_strike, neighbors_each))
    log.info("Neighbors: weekday=%s ±%s, wanted_count=%s", today_date.weekday(), neighbors_each, len(wanted))

    for c in ("CE.changeinOpenInterest", "PE.changeinOpenInterest"):
        if c not in df_all.columns:
            df_all[c] = None

    df = df_all[["strikePrice", "CE.changeinOpenInterest", "PE.changeinOpenInterest"]].rename(
        columns={
            "strikePrice": "Strike",
            "CE.changeinOpenInterest": "Call Chg OI",
            "PE.changeinOpenInterest": "Put Chg OI",
        }
    )
    df = df[df["Strike"].isin(wanted)].copy()
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce").astype("Int64")
    df["Call Chg OI"] = pd.to_numeric(df["Call Chg OI"], errors="coerce")
    df["Put Chg OI"]  = pd.to_numeric(df["Put Chg OI"],  errors="coerce")
    df = df.sort_values("Strike").reset_index(drop=True)

    call_sum = float(df["Call Chg OI"].sum(skipna=True))
    put_sum  = float(df["Put Chg OI"].sum(skipna=True))
    denom = call_sum + put_sum
    if denom == 0:
        puts_pct = calls_pct = imbalance_pct = 0.0
    else:
        puts_pct = (put_sum / denom) * 100.0
        calls_pct = (call_sum / denom) * 100.0
        imbalance_pct = puts_pct - calls_pct

    suggestion = "NO SIGNAL"
    if abs(imbalance_pct) > IMBALANCE_TRIGGER:
        suggestion = "BUY PUT" if imbalance_pct < 0 else "BUY CALL"

    updated_str = now_ist().strftime("%Y-%m-%d %H:%M:%S")

    df.insert(0, "ATM", atm_strike)
    df.insert(0, "Expiry", expiry)
    df.insert(0, "Updated", updated_str)
    df["Put Σ Chg OI"]  = put_sum
    df["Call Σ Chg OI"] = call_sum
    df["PUTS %"]        = round(puts_pct, 2)
    df["CALLS %"]       = round(calls_pct, 2)
    df["Imbalance %"]   = round(imbalance_pct, 2)
    df["Suggestion"]    = suggestion

    # re-read store to show latest status/base (in case TV loop upgraded mid-build)
    latest_store = load_atm_store()
    atm_status_disp = latest_store.get("atm_status", "unknown")
    base_value_disp = latest_store.get("base_value", None)

    meta = {
        "neighbors_each": neighbors_each,
        "underlying": float(records.get("underlyingValue", 0.0)),
        "call_sum": call_sum,
        "put_sum": put_sum,
        "puts_pct": puts_pct,
        "calls_pct": calls_pct,
        "imbalance_pct": imbalance_pct,
        "suggestion": suggestion,
        "expiry": expiry,
        "atm": atm_strike,
        "updated": updated_str,
        "atm_status": atm_status_disp,
        "base_value": base_value_disp,
    }
    log.info("Imbalance: put_sum=%.0f call_sum=%.0f imb=%.2f%% sugg=%s; ATM=%s (%s)",
             put_sum, call_sum, imbalance_pct, suggestion, atm_strike, atm_status_disp)
    return df, meta

# ---------------- Memory store & loops ----------------
class StoreMem:
    def __init__(self):
        self.lock = threading.Lock()
        self.df_opt: pd.DataFrame | None = None
        self.meta_opt: dict = {}
        self.last_opt: dt.datetime | None = None

        self.vwap_latest: float | None = None
        self.vwap_df15: pd.DataFrame | None = None
        self.last_tv: dt.datetime | None = None

        self.vwap_alert: str = "NO ALERT"
        self.last_alert_key: str = ""

def option_chain_loop(mem: StoreMem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            raw = fetch_raw_option_chain()
            df, meta = build_df_with_imbalance(raw, {})
            if not df.empty:
                with mem.lock:
                    mem.df_opt = df.copy()
                    mem.meta_opt = dict(meta)
                    mem.last_opt = now_ist()
                try:
                    df.to_csv(CSV_PATH, index=False)
                except Exception as e:
                    log.error("Write CSV failed: %s", e)
                log.info("[OC] wrote %d rows", len(df))
            else:
                log.warning("[OC] empty dataframe this cycle")
        except Exception as e:
            log.exception("OptionChain loop error: %s", e)
        time.sleep(FETCH_EVERY_SECONDS)

def write_vwap_files(stamp: str, vwap_latest: float | None, spot: float | None, suggestion: str):
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        v = f"{vwap_latest:.2f}" if vwap_latest is not None else "NA"
        s = f"{float(spot):.2f}" if spot is not None else "NA"
        VWAP_NOW_TXT.write_text(f"{stamp} IST | VWAP15m={v} | Spot={s} | Signal={suggestion}\n")
        header_needed = not VWAP_LOG_CSV.exists()
        with VWAP_LOG_CSV.open("a", encoding="utf-8") as f:
            if header_needed:
                f.write("timestamp_ist,vwap15m,spot,signal\n")
            f.write(f"{stamp},{v},{s},{suggestion}\n")
    except Exception as e:
        log.error("VWAP file write failed: %s", e)

# -----------------------------------------------------------------------------
# TV‑loop: pulls 1‑minute NIFTY data, upgrades ATM instantly, refreshes imbalance
# -----------------------------------------------------------------------------
def tradingview_loop(mem: StoreMem):
    """
    1. Pull 1‑minute candles from TradingView every TV_FETCH_SECONDS.
    2. If a fresh 09:09 IST price appears, update ATM → store → *immediately*
       rebuild the option‑chain dataframe so imbalance / suggestion stay current.
    3. Compute session VWAP (15‑minute cumulative) for the dashboard + alert.
    4. Write a one‑line status file and append to the rolling VWAP CSV log.
    """

    while True:
        try:
            # ---- 1) Get latest 1‑minute candles --------------------------------
            df1 = fetch_tv_1m_session()                       # retry logic inside

            # ---- 2) Instant ATM upgrade if 09:09 available --------------------
            px909 = price_at_0909(df1) if df1 is not None else None
            if px909 and px909 > 0:
                base_val   = float(px909)
                atm_guess  = round_to_50(base_val)

                store = load_atm_store()
                needs_upgrade = (
                    store.get("date")        != today_str() or
                    store.get("atm_status")  != "captured-0909" or
                    int(store.get("atm_strike", 0)) != atm_guess
                )

                if needs_upgrade:
                    update_store_atm(atm_guess, base_val, "captured-0909")
                    log.info("ATM upgraded to %s (base %.2f) by TV‑loop", atm_guess, base_val)

                    # ---- 2a) Recalculate imbalance right away ----------------
                    raw_now = fetch_raw_option_chain()
                    df_now, meta_now = build_df_with_imbalance(raw_now, {})
                    if not df_now.empty:
                        with mem.lock:
                            mem.df_opt   = df_now.copy()
                            mem.meta_opt = dict(meta_now)
                            mem.last_opt = now_ist()
                        try:
                            df_now.to_csv(CSV_PATH, index=False)
                        except Exception as e:
                            log.error("CSV write failed (TV‑trigger): %s", e)
                        log.info("Imbalance refreshed immediately after ATM upgrade")

            # ---- 3) VWAP (15‑minute session cumulative) -----------------------
            vwap_latest, df15 = compute_session_vwap_15m(df1)
            with mem.lock:
                mem.last_tv     = now_ist()
                mem.vwap_latest = vwap_latest
                mem.vwap_df15   = df15

            # ---- 4) Evaluate VWAP alert ---------------------------------------
            with mem.lock:
                meta = mem.meta_opt or {}
                spot = meta.get("underlying")
                sugg = meta.get("suggestion", "NO SIGNAL")

            alert = "NO ALERT"
            if (
                vwap_latest is not None and
                spot is not None and
                sugg in ("BUY CALL", "BUY PUT") and
                abs(float(spot) - float(vwap_latest)) <= VWAP_TOLERANCE_PTS
            ):
                alert = f"{sugg} (spot near VWAP ±{VWAP_TOLERANCE_PTS})"

            with mem.lock:
                mem.vwap_alert = alert

            # ---- 5) Persist VWAP snapshot & log line --------------------------
            stamp = now_ist().strftime("%Y-%m-%d %H:%M:%S")
            write_vwap_files(stamp, vwap_latest, spot, sugg)

            log.info("[TV] vwap=%s alert=%s",
                     f"{vwap_latest:.2f}" if vwap_latest is not None else "None",
                     alert)

        except Exception as e:
            # Keep the loop alive even on unexpected errors
            with mem.lock:
                mem.last_tv     = now_ist()
                mem.vwap_latest = None
            log.exception("TradingView loop error: %s", e)

        time.sleep(TV_FETCH_SECONDS)


@st.cache_resource
def start_background() -> StoreMem:
    mem = StoreMem()
    threading.Thread(target=option_chain_loop, args=(mem,), daemon=True, name="OC-Loop").start()
    threading.Thread(target=tradingview_loop, args=(mem,), daemon=True, name="TV-Loop").start()
    return mem

# ---------------- UI helpers ----------------
def play_beep_once_on_new_alert(mem: StoreMem, alert_text: str):
    key = f"{today_str()}|{alert_text}"
    if alert_text != "NO ALERT" and key != mem.last_alert_key:
        st.markdown(
            f"""
            <audio autoplay>
              <source src="data:audio/wav;base64,{BEEP_WAV_B64}" type="audio/wav">
            </audio>
            """,
            unsafe_allow_html=True
        )
        mem.last_alert_key = key

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="NIFTY ΔOI Imbalance + TV VWAP Alert", layout="wide")
st_autorefresh(interval=AUTOREFRESH_MS, key="nifty_autorefresh")

# Sidebar thresholds
with st.sidebar:
    st.header("Settings")
    VWAP_tol = st.number_input("VWAP tolerance (pts)", value=float(VWAP_TOLERANCE_PTS), step=1.0)
    IMB_thr  = st.number_input("Imbalance trigger (%)", value=float(IMBALANCE_TRIGGER), step=1.0)
    st.caption(f"Logs: `{LOG_PATH}`")
    st.caption(f"Latest VWAP: `{VWAP_NOW_TXT}`")

    st.divider()
    st.subheader("Manual ATM override")
    man_atm = st.number_input("Set ATM strike (multiple of 50)", min_value=0, step=50, value=0)
    if st.button("Apply ATM override"):
        if man_atm > 0:
            update_store_atm(int(man_atm), float(man_atm), "manual-override")
            st.success(f"ATM overridden to {int(man_atm)}")
        else:
            st.warning("Enter a positive strike.")

    if st.button("Show last 80 log lines"):
        try:
            lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[-80:]
            st.code("\n".join(lines))
        except Exception as e:
            st.error(f"Could not read log: {e}")

mem = start_background()

with mem.lock:
    df_live = None if mem.df_opt is None else mem.df_opt.copy()
    meta = dict(mem.meta_opt)
    last_opt = mem.last_opt
    vwap_latest = mem.vwap_latest
    last_tv = mem.last_tv
    vwap_alert = mem.vwap_alert

st.title("NIFTY Change in OI — Imbalance + VWAP Alert (TradingView)")

# Status row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last OC pull", last_opt.strftime("%H:%M:%S") if last_opt else "—")
c2.metric("Last TV pull", last_tv.strftime("%H:%M:%S") if last_tv else "—")
c3.metric("Spot (underlying)", f"{meta.get('underlying', float('nan')):,.2f}" if meta else "—")
c4.metric("VWAP (15m session)", f"{vwap_latest:,.2f}" if vwap_latest else "—")
c5.metric("VWAP tolerance", f"±{VWAP_tol:.0f} pts")

if df_live is None or df_live.empty:
    st.warning("Waiting for first successful option-chain fetch…")
    st.stop()

expiry = meta.get("expiry", str(df_live["Expiry"].iloc[0]))
atm_strike = meta.get("atm", int(df_live["ATM"].iloc[0]))
atm_status = meta.get("atm_status", "unknown")
base_value = meta.get("base_value", None)
updated_str = meta.get("updated", str(df_live["Updated"].iloc[0]))
imbalance_pct = meta.get("imbalance_pct", float(df_live.get("Imbalance %", pd.Series([0])).iloc[0]))
suggestion = meta.get("suggestion", str(df_live.get("Suggestion", pd.Series(["NO SIGNAL"])).iloc[0]))
neighbors_each = meta.get("neighbors_each", 1)
call_sum = meta.get("call_sum", float(df_live["Call Σ Chg OI"].iloc[0]))
put_sum  = meta.get("put_sum",  float(df_live["Put Σ Chg OI"].iloc[0]))
puts_pct = meta.get("puts_pct", float(df_live["PUTS %"].iloc[0]))
calls_pct= meta.get("calls_pct",float(df_live["CALLS %"].iloc[0]))
spot     = meta.get("underlying", None)

# Apply sidebar thresholds for display
imbalance_ok = abs(imbalance_pct) > IMB_thr
vwap_ok = (vwap_latest is not None and spot is not None and abs(float(spot) - float(vwap_latest)) <= VWAP_tol)
combined_alert = "NO ALERT"
if suggestion in ("BUY CALL", "BUY PUT") and imbalance_ok and vwap_ok:
    combined_alert = f"{suggestion} (spot near VWAP ±{VWAP_tol})"

# Banner + sound
if combined_alert != "NO ALERT":
    st.success(f"VWAP ALERT: **{combined_alert}**", icon="✅")
    with mem.lock:
        play_beep_once_on_new_alert(mem, combined_alert)
else:
    st.info("No VWAP alert yet. Needs active BUY signal and |Spot−VWAP| within tolerance.", icon="ℹ️")

st.subheader(f"Expiry: {expiry}")
base_disp = f"{base_value:,.2f}" if isinstance(base_value, (int, float)) else "—"
st.caption(
    f"Updated: **{updated_str} IST** • ATM: **{atm_strike}** (**{atm_status}**, base={base_disp}) • "
    f"Neighbors each side (weekday rule): **{neighbors_each}**"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("PUT Σ Chg OI", f"{put_sum:,.0f}")
k2.metric("CALL Σ Chg OI", f"{call_sum:,.0f}")
k3.metric("PUTS %", f"{puts_pct:,.2f}%")
k4.metric("CALLS %", f"{calls_pct:,.2f}%")
k5.metric("Imbalance (PUTS − CALLS)", f"{imbalance_pct:,.2f}%")

# VWAP/Spot caption
if vwap_latest is not None and spot is not None:
    st.caption(f"VWAP15m: **{vwap_latest:,.2f}**  •  Spot: **{spot:,.2f}**  •  Diff: **{spot - vwap_latest:+.2f}**")
else:
    st.caption("VWAP or Spot not available yet. Check logs if this persists.")

st.dataframe(
    df_live[["Updated","Expiry","ATM","Strike","Call Chg OI","Put Chg OI",
             "Put Σ Chg OI","Call Σ Chg OI","PUTS %","CALLS %","Imbalance %","Suggestion"]]
      .sort_values("Strike"),
    use_container_width=True
)

plot_df = df_live.melt(id_vars=["Strike"],
                       value_vars=["Call Chg OI", "Put Chg OI"],
                       var_name="Side", value_name="Chg OI").sort_values("Strike")
title = f"ΔOI by Strike (ATM {atm_strike}, ±{neighbors_each}) • Imbalance {imbalance_pct:,.2f}% → {suggestion}"
fig = px.bar(plot_df, x="Strike", y="Chg OI", color="Side", barmode="group", text="Chg OI", title=title)
fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
fig.update_layout(xaxis=dict(type="category"), margin=dict(t=80, r=20, l=20, b=40))
st.plotly_chart(fig, use_container_width=True)
