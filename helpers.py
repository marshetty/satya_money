# helpers.py
# ---------------------------------------------------------------------------
# Pure-logic helpers shared by dashboard.py (no Streamlit or UI code here)
# ---------------------------------------------------------------------------

import datetime as dt, time, math, random, json, logging, pathlib
import pandas as pd, requests, certifi, warnings, os

log = logging.getLogger("nifty_app")       # dashboard initializes the handler

SYMBOL = "NIFTY"
API_URL = "https://www.nseindia.com/api/option-chain-indices"
IST   = dt.timezone(dt.timedelta(hours=5, minutes=30))

BASE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"https://www.nseindia.com/option-chain?symbol={SYMBOL}",
}
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# ---------- simple helpers --------------------------------------------------
def now_ist()  -> dt.datetime: return dt.datetime.now(IST)
def today_str()               : return now_ist().strftime("%Y%m%d")

def round_to_50(x: float) -> int:
    """Round a float to the nearest multiple of 50."""
    return int(round(x / 50.0) * 50)

def neighbors_by_weekday(d: dt.date) -> int:
    """Fri/Sat/Sun ±5, Mon ±4, Tue ±3, Wed ±2, Thu ±1."""
    return {0:4, 1:3, 2:2, 3:1, 4:5, 5:5, 6:5}.get(d.weekday(), 3)

# ---------- NSE option-chain fetch ------------------------------------------
def new_session():
    try:
        import cloudscraper
        warnings.filterwarnings("ignore", category=UserWarning, module="cloudscraper")
        s = cloudscraper.create_scraper(delay=8, browser={'browser':'chrome','platform':'windows'})
    except ModuleNotFoundError:
        import requests as _rq
        s = _rq.Session()
    s.headers.update(BASE_HEADERS)
    try: s.get(f"https://www.nseindia.com/option-chain?symbol={SYMBOL}", timeout=8)
    except Exception: pass
    return s

def fetch_raw_option_chain(retries: int = 6) -> dict | None:
    sess = new_session()
    for i in range(retries):
        try:
            r = sess.get(API_URL, params={"symbol": SYMBOL}, timeout=10)
            if r.status_code == 200:
                raw = r.json()
                if "records" in raw and "data" in raw["records"]:
                    log.info("OC fetch OK (try %d)", i+1)
                    return raw
            log.warning("OC HTTP %s (try %d)", r.status_code, i+1)
        except Exception as e:
            log.warning("OC fetch exception (try %d): %s", i+1, e)
        time.sleep(2)
    log.error("OC fetch failed after retries.")
    return None

def pick_current_week_expiry(raw: dict) -> str | None:
    today = now_ist().date()
    parsed=[]
    for s in raw.get("records",{}).get("expiryDates",[]):
        try: parsed.append((s, dt.datetime.strptime(s,"%d-%b-%Y").date()))
        except: pass
    if not parsed: return None
    future=[p for p in parsed if p[1]>=today]
    return (min(future,key=lambda x:x[1]) if future else min(parsed,key=lambda x:x[1]))[0]

# ---------- imbalance builder -----------------------------------------------
def build_df_with_imbalance(raw: dict, max_neighbors: int = 20):
    """
    Takes raw NSE option-chain JSON → returns
      df   : strikes around ATM with ΔOI & stats
      meta : dictionary summary (ΣPUT, ΣCALL, imbalance %, suggestion …)
    """
    if not raw: return pd.DataFrame(), {}

    expiry = pick_current_week_expiry(raw)
    if not expiry: return pd.DataFrame(), {}

    records = raw["records"]
    rows = [x for x in records["data"] if x.get("expiryDate")==expiry]
    if not rows: return pd.DataFrame(), {}

    df_all = pd.json_normalize(rows)
    strikes_sorted = sorted({int(v) for v in df_all["strikePrice"].dropna().astype(int)})

    # ---------- choose ATM ---------------------------------------------------
    underlying = float(records.get("underlyingValue",0.0))
    atm_guess  = round_to_50(underlying)
    atm = atm_guess if atm_guess in strikes_sorted else min(strikes_sorted, key=lambda x:abs(x-underlying))

    # ---------- neighbor slicing --------------------------------------------
    neighbors_each = min(neighbors_by_weekday(now_ist().date()), max_neighbors)
    idx = strikes_sorted.index(atm)
    start = max(0, idx-neighbors_each)
    end   = min(len(strikes_sorted)-1, idx+neighbors_each)
    wanted = set(strikes_sorted[start:end+1])

    for col in ("CE.changeinOpenInterest", "PE.changeinOpenInterest"):
        if col not in df_all.columns: df_all[col]=0

    df = df_all[["strikePrice","CE.changeinOpenInterest","PE.changeinOpenInterest"]]
    df = df[df["strikePrice"].isin(wanted)].copy()
    df.columns = ["Strike","Call Chg OI","Put Chg OI"]
    df = df.sort_values("Strike").reset_index(drop=True)

    call_sum = float(df["Call Chg OI"].sum(skipna=True))
    put_sum  = float(df["Put Chg OI"].sum(skipna=True))
    denom = call_sum + put_sum
    if denom == 0: puts_pct=calls_pct=imb=0.0
    else:
        puts_pct = put_sum/denom*100
        calls_pct= call_sum/denom*100
        imb      = puts_pct - calls_pct

    suggestion="NO SIGNAL"
    if abs(imb)>30:
        suggestion = "BUY PUT" if imb<0 else "BUY CALL"

    df.insert(0,"ATM",atm)
    df.insert(0,"Expiry",expiry)
    df["Put Σ Chg OI"]  = put_sum
    df["Call Σ Chg OI"] = call_sum
    df["PUTS %"]        = round(puts_pct,2)
    df["CALLS %"]       = round(calls_pct,2)
    df["Imbalance %"]   = round(imb,2)
    df["Suggestion"]    = suggestion
    df["Updated"]       = now_ist().strftime("%Y-%m-%d %H:%M:%S")

    meta = dict(underlying=underlying, call_sum=call_sum, put_sum=put_sum,
                puts_pct=puts_pct, calls_pct=calls_pct, imbalance_pct=imb,
                suggestion=suggestion, expiry=expiry, atm=atm,
                neighbors_each=neighbors_each)
    return df, meta
