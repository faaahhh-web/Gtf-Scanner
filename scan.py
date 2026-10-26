"""GTF scanner -> results.json (shown by index.html on GitHub Pages).
Env: ANGEL_API_KEY ANGEL_CLIENT_ID ANGEL_PIN ANGEL_TOTP_SECRET ; optional CAPITAL, RISK_PCT, ONLY
"""
import os, json, datetime as dt
import gtf_core as g
import angel_data as ad

OUT = os.environ.get("GTF_OUT", "results.json")


def scan_symbol(u, raw, risk_amt):
    sigs = []
    cmp_ = raw["15m"][-1].c if raw["15m"] and raw["15m"][-1].t >= raw["1D"][-1].t else raw["1D"][-1].c
    for mode, (htf, itf, ltf) in ad.MODES.items():
        series = {n: ad.drop_incomplete(ad.get_tf(raw, n), n) for n in (htf, itf, ltf)}
        for s in g.evaluate(u["symbol"], series[ltf], series[itf], series[htf], cmp_, u["lot"], risk_amt):
            s["mode"] = mode
            sigs.append(s)
    return sigs


def run():
    capital = float(os.environ.get("CAPITAL", 100000))
    risk_pct = float(os.environ.get("RISK_PCT", 1.0))
    risk_amt = capital * risk_pct / 100.0
    sess = ad.login()
    uni = ad.load_universe()
    only = os.environ.get("ONLY")
    if only:
        keep = {x.strip().upper() for x in only.split(",")}
        uni = [u for u in uni if u["symbol"] in keep]
    sigs, errs = [], []
    for u in uni:
        try:
            raw = {k: ad.candles(sess, u["token"], iv, days) for k, (iv, days) in ad.FETCH.items()}
            if any(len(v) < 60 for v in raw.values()):
                errs.append(u["symbol"]); continue
            sigs += scan_symbol(u, raw, risk_amt)
        except Exception as e:
            errs.append(u["symbol"]); print("ERR", u["symbol"], e)
    sigs.sort(key=lambda s: (-s["score"], s["symbol"]))
    out = {"updated": dt.datetime.now(ad.IST).strftime("%d %b %Y %H:%M IST"), "scanned": len(uni),
           "errors": errs, "capital": capital, "risk_pct": risk_pct, "signals": sigs}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(out, open(OUT, "w"), default=str)
    print(f"{len(uni)} symbols, {len(sigs)} signals, {len(errs)} errors")


if __name__ == "__main__":
    run()
