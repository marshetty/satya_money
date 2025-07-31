# dashboard.py
# ---------------------------------------------------------------------------
# Streamlit UI + background loops (imports all logic from helpers.py)
# ---------------------------------------------------------------------------

import os, time, threading, pathlib, logging, datetime as dt
import pandas as pd, streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
from helpers import (
    round_to_50, build_df_with_imbalance, neighbors_by_weekday,
    fetch_raw_option_chain, now_ist, today_str, pick_current_week_expiry
)
from helpers import log, SYMBOL, IST                       # reuse logger & constants
from helpers import new_session                            # if you want it elsewhere

# ============== USER SETTINGS ==============
FETCH_EVERY_SECONDS  = 60
TV_FETCH_SECONDS     = 60
AUTOREFRESH_MS       = 10_000
VWAP_LOOKBACK_MIN    = 15
VWAP_TOLERANCE_PTS   = 15.0
# ===========================================

OUT_DIR = pathlib.Path.home()/".nifty_app"
CSV_PATH = OUT_DIR/"nifty_oc.csv"

# ---------- simple in-memory store ---------
class Mem:
    df_opt=None; meta_opt={}; last_oc=None
    vwap=None;  last_tv=None; alert="NO ALERT"
    lock=threading.Lock()
mem=Mem()

# ---------- TradingView VWAP helpers -------
def tv_login():
    from tvDatafeed import TvDatafeed
    tv=TvDatafeed(username=os.getenv("TV_USERNAME"),password=os.getenv("TV_PASSWORD"))
    return tv

def fetch_tv_1m():
    from tvDatafeed import Interval
    tv=tv_login()
    df=tv.get_hist("NIFTY","NSE",Interval.in_1_minute,n_bars=1300)
    if df.index.tz is None:
        df.index=df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df.index=df.index.tz_convert("Asia/Kolkata")
    return df

def rolling_vwap(df, lookback):                               # typical-price VWAP
    if df.empty: return None
    df=df[df.index >= df.index.max()-dt.timedelta(minutes=lookback)]
    tp=((df["high"]+df["low"]+df["close"])/3).astype(float)
    vol=df["volume"].fillna(0).astype(float)
    if vol.sum()==0: return float(tp.mean())
    return float((tp*vol).sum()/vol.sum())

# ---------- background loops --------------
def oc_loop():
    while True:
        raw=fetch_raw_option_chain()
        df,meta=build_df_with_imbalance(raw)
        if not df.empty:
            with mem.lock:
                mem.df_opt,mem.meta_opt,mem.last_oc=df,meta,now_ist()
            df.to_csv(CSV_PATH,index=False)
        time.sleep(FETCH_EVERY_SECONDS)

def tv_loop():
    while True:
        try:
            df1=fetch_tv_1m()
            vwap=rolling_vwap(df1,VWAP_LOOKBACK_MIN)
            with mem.lock:
                mem.vwap,mem.last_tv=vwap,now_ist()
                spot=mem.meta_opt.get("underlying")
                sugg=mem.meta_opt.get("suggestion","NO SIGNAL")
            if vwap and spot and sugg.startswith("BUY") and abs(spot-vwap)<=VWAP_TOLERANCE_PTS:
                with mem.lock: mem.alert=f"{sugg} (spot near VWAP)"
            else:
                with mem.lock: mem.alert="NO ALERT"
        except Exception as e:
            log.error("TV loop error: %s",e)
        time.sleep(TV_FETCH_SECONDS)

# ---------- kick off threads --------------
@st.cache_resource
def start_threads():
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    threading.Thread(target=oc_loop,daemon=True,name="OC").start()
    threading.Thread(target=tv_loop,daemon=True,name="TV").start()
start_threads()

# ----------------- UI ---------------------
st.set_page_config("NIFTY ΔOI + Rolling VWAP",layout="wide")
st_autorefresh(AUTOREFRESH_MS,key="refresh")

with mem.lock:
    df=mem.df_opt.copy() if mem.df_opt is not None else pd.DataFrame()
    meta=mem.meta_opt.copy()
    vwap=mem.vwap; alert=mem.alert
    last_oc=mem.last_oc; last_tv=mem.last_tv

st.title("NIFTY Change-in-OI Imbalance + Rolling VWAP Alert")

c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Last OC", last_oc.strftime("%H:%M:%S") if last_oc else "—")
c2.metric("Last TV", last_tv.strftime("%H:%M:%S") if last_tv else "—")
c3.metric("Spot",  f"{meta.get('underlying',0):,.2f}" if meta else "—")
c4.metric(f"VWAP({VWAP_LOOKBACK_MIN}m)", f"{vwap:,.2f}" if vwap else "—")
c5.metric("Alert", alert)

if alert!="NO ALERT":
    st.success(alert)

if df.empty:
    st.warning("Waiting for option-chain …"); st.stop()

st.dataframe(df,use_container_width=True)

plotdf=df.melt(id_vars="Strike",
               value_vars=["Call Chg OI","Put Chg OI"],
               var_name="Side",value_name="ΔOI")
fig=px.bar(plotdf,x="Strike",y="ΔOI",color="Side",barmode="group",
           title=f"ΔOI around ATM {meta['atm']} ({meta['expiry']})")
fig.update_traces(texttemplate="%{y:,}",textposition="outside")
st.plotly_chart(fig,use_container_width=True)
