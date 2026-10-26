"""Angel One SmartAPI data layer: F&O universe, candles, resampling to GTF timeframes."""
import os, time, json, datetime as dt
import urllib.request
from gtf_core import Bar

MASTER_URLS = [
    "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json",
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
]
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
OPEN = dt.time(9, 15)
CLOSE = dt.time(15, 30)

# income mode -> (HTF, ITF, LTF) per the notes' timeframe table
MODES = {
    "daily":     ("1D", "75m", "15m"),
    "monthly":   ("1M", "1W", "1D"),
    "quarterly": ("1Q", "1M", "1W"),   # EXTRAPOLATED: not in the notes' table
}
INDEX_TOKENS = {"NIFTY": "99926000", "BANKNIFTY": "99926009"}   # verify once against the master


def login():
    import pyotp
    from SmartApi import SmartConnect
    s = SmartConnect(api_key=os.environ["ANGEL_API_KEY"])
    d = s.generateSession(os.environ["ANGEL_CLIENT_ID"], os.environ["ANGEL_PIN"],
                          pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now())
    if not d.get("status"):
        raise RuntimeError(f"Angel login failed: {d.get('message')}")
    return s


def load_universe(include_indices=True):
    """F&O underlyings: nearest-expiry futures give lot size; NSE cash token gives the chart."""
    m, last = None, None
    for url in MASTER_URLS:
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    raw = r.read()
                m = json.loads(raw)
                break
            except Exception as e:      # incl. 404, truncated downloads (IncompleteRead), bad JSON
                last = f"{url} -> {e!r}"
                print("instrument list attempt", attempt + 1, "failed:", last)
                time.sleep(3 * (attempt + 1))
        if m is not None:
            break
    if m is None:
        raise RuntimeError(f"Could not download Angel instrument list: {last}")
    futs = {}
    for x in m:
        if x["exch_seg"] == "NFO" and x["instrumenttype"] in ("FUTSTK", "FUTIDX"):
            exp = dt.datetime.strptime(x["expiry"], "%d%b%Y").date()
            if exp < dt.date.today():
                continue
            k = x["name"]
            if k not in futs or exp < futs[k][0]:
                futs[k] = (exp, int(x["lotsize"]), x["instrumenttype"])
    cash = {x["name"]: x["token"] for x in m
            if x["exch_seg"] == "NSE" and x["symbol"].endswith("-EQ")}
    out = []
    for name, (exp, lot, typ) in futs.items():
        if typ == "FUTSTK" and name in cash:
            out.append({"symbol": name, "token": cash[name], "lot": lot})
        elif typ == "FUTIDX" and include_indices and name in INDEX_TOKENS:
            out.append({"symbol": name, "token": INDEX_TOKENS[name], "lot": lot})
    return sorted(out, key=lambda x: x["symbol"])


_last = [0.0]
def _throttle(gap=0.4):
    w = gap - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    _last[0] = time.time()


def candles(sess, token, interval, days, exch="NSE", retries=3):
    to = dt.datetime.now(IST)
    frm = to - dt.timedelta(days=days)
    p = {"exchange": exch, "symboltoken": token, "interval": interval,
         "fromdate": frm.strftime("%Y-%m-%d %H:%M"), "todate": to.strftime("%Y-%m-%d %H:%M")}
    for a in range(retries):
        _throttle()
        try:
            d = sess.getCandleData(p)
            if d and d.get("status") and d.get("data"):
                return [Bar(dt.datetime.fromisoformat(r[0]), *map(float, r[1:6])) for r in d["data"]]
        except Exception:
            pass
        time.sleep(1.5 * (a + 1))
    return []


# ---------- resampling ----------
def _session_group(t, minutes):
    """Index of the 'minutes'-long bucket from 9:15 for an intraday bar start time."""
    off = (t.hour * 60 + t.minute) - (OPEN.hour * 60 + OPEN.minute)
    return t.date(), off // minutes


def resample_intraday(bars, minutes):
    out, cur, key = [], None, None
    for b in bars:
        k = _session_group(b.t, minutes)
        if k != key:
            if cur:
                out.append(cur)
            cur, key = Bar(b.t, b.o, b.h, b.l, b.c, b.v), k
        else:
            cur.h, cur.l, cur.c, cur.v = max(cur.h, b.h), min(cur.l, b.l), b.c, cur.v + b.v
    if cur:
        out.append(cur)
    return out


def resample_period(daily, kind):
    out, cur, key = [], None, None
    for b in daily:
        d = b.t.date()
        k = (d.isocalendar()[:2] if kind == "1W" else
             (d.year, (d.month - 1) // 3) if kind == "1Q" else (d.year, d.month))
        if k != key:
            if cur:
                out.append(cur)
            cur, key = Bar(b.t, b.o, b.h, b.l, b.c, b.v), k
        else:
            cur.h, cur.l, cur.c, cur.v = max(cur.h, b.h), min(cur.l, b.l), b.c, cur.v + b.v
    if cur:
        out.append(cur)
    return out


def tf_minutes(tf):
    return {"5m": 5, "15m": 15, "75m": 75, "125m": 125}.get(tf)


def drop_incomplete(bars, tf, now=None):
    """Remove the last bar if its period has not finished yet."""
    if not bars:
        return bars
    now = now or dt.datetime.now(IST)
    last = bars[-1]
    m = tf_minutes(tf)
    if m:
        end = last.t + dt.timedelta(minutes=m)
        cap = dt.datetime.combine(last.t.date(), CLOSE, tzinfo=last.t.tzinfo)
        end = min(end, cap)
    elif tf == "1D":
        end = dt.datetime.combine(last.t.date(), CLOSE, tzinfo=last.t.tzinfo)
    elif tf == "1W":
        fri = last.t.date() + dt.timedelta(days=4 - last.t.weekday())
        end = dt.datetime.combine(fri, CLOSE, tzinfo=last.t.tzinfo)
    elif tf == "1Q":
        qm = ((last.t.month - 1) // 3 + 1) * 3
        base = last.t.date().replace(month=qm, day=28)
        nxt = (base + dt.timedelta(days=4)).replace(day=1)
        end = dt.datetime.combine(nxt, OPEN, tzinfo=last.t.tzinfo)
    else:  # 1M
        nxt = (last.t.date().replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        end = dt.datetime.combine(nxt, OPEN, tzinfo=last.t.tzinfo)
    return bars[:-1] if now < end else bars


def get_tf(raw, tf):
    """raw = dict of base series {'5m','15m','1D'} -> requested GTF timeframe."""
    if tf in ("5m", "15m", "1D"):
        return raw[tf]
    if tf == "75m":
        return resample_intraday(raw["15m"], 75)
    if tf == "125m":
        return resample_intraday(raw["5m"], 125)
    return resample_period(raw["1D"], tf) if tf in ("1M", "1Q") else resample_period(raw["1D"], tf)


# one fetch per symbol serves all three modes
FETCH = {"15m": ("FIFTEEN_MINUTE", 120), "1D": ("ONE_DAY", 1900)}   # Angel caps daily history (~2000 days)
# hourly needs the 75m HTF built from 15m; 5m is LTF, so both bases are fetched above.
