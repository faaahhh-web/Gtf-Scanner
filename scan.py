"""GTF scanner -> docs/results.json (shown by docs/index.html on GitHub Pages).
Env: ANGEL_API_KEY ANGEL_CLIENT_ID ANGEL_PIN ANGEL_TOTP_SECRET ; optional CAPITAL, RISK_PCT, ONLY
"""
import os, json, datetime as dt
import gtf_core as g
import angel_data as ad

OUT = os.environ.get("GTF_OUT", "docs/results.json")


DIAG = {m: {"ltf_zones": 0, "htf_zones": 0, "signals": 0} for m in ad.MODES}


def scan_symbol(u, raw, risk_amt):
    sigs = []
    cmp_ = raw["15m"][-1].c if raw["15m"] and raw["15m"][-1].t >= raw["1D"][-1].t else raw["1D"][-1].c
    for mode, (htf, itf, ltf) in ad.MODES.items():
        series = {n: ad.drop_incomplete(ad.get_tf(raw, n), n) for n in (htf, itf, ltf)}
        DIAG[mode]["ltf_zones"] += sum(1 for z in g.build_zones(g.clean(series[ltf])) if g.valid(z) and z.core >= g.CFG["QUALITY_MIN_CORE"])
        DIAG[mode]["htf_zones"] += sum(1 for z in g.build_zones(g.clean(series[htf])) if g.valid(z) and z.score >= g.CFG["MIN_SCORE"])
        for s in g.evaluate(u["symbol"], series[ltf], series[itf], series[htf], cmp_, u["lot"], risk_amt, align=(mode != "quarterly")):
            s["mode"] = mode
            DIAG[mode]["signals"] += 1
            sigs.append(s)
    return sigs


def scan_breakouts(u, raw):
    d1 = ad.drop_incomplete(ad.get_tf(raw, "1D"), "1D")
    w1 = ad.drop_incomplete(ad.get_tf(raw, "1W"), "1W")
    m1 = ad.drop_incomplete(ad.get_tf(raw, "1M"), "1M")
    cmp_ = raw["15m"][-1].c if raw["15m"] and raw["15m"][-1].t >= raw["1D"][-1].t else raw["1D"][-1].c
    return g.breakout_setups(u["symbol"], d1, w1, m1, cmp_)


def scan_board(u, raw, risk_amt):
    cmp_ = raw["15m"][-1].c if raw["15m"] and raw["15m"][-1].t >= raw["1D"][-1].t else raw["1D"][-1].c
    tfb = {tf: ad.drop_incomplete(ad.get_tf(raw, tf), tf) for tf in g.BOARD_TFS}
    return g.zone_board(u["symbol"], tfb, raw["1D"], cmp_, u["lot"], risk_amt)


def bucket_key(r):
    return f'{"demand" if r["kind"] == "D" else "supply"}|{r["state"]}|%s|{r["tf"]}'


def update_history(path, date, board, trades):
    hist = {"dates": {}}
    try:
        hist = json.load(open(path))
    except Exception:
        pass
    b, t = {}, {}
    for r in board:
        k = bucket_key(r)
        b.setdefault(k % "standard", []).append(r["symbol"])
        if r["signal"]:
            b.setdefault(k % "signal", []).append(r["symbol"])
    for x in trades:
        t.setdefault(f'{"demand" if x["kind"] == "D" else "supply"}|{x["type"]}', []).append(x["symbol"])
    hist["dates"][date] = {"b": b, "t": t}
    for d in sorted(hist["dates"])[:-60]:          # keep the last 60 trading days
        del hist["dates"][d]
    json.dump(hist, open(path, "w"), separators=(",", ":"))


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
    sigs, bos, board, trades, errs = [], [], [], [], []
    for u in uni:
        try:
            raw = {k: ad.candles(sess, u["token"], iv, days) for k, (iv, days) in ad.FETCH.items()}
            if any(len(v) < 60 for v in raw.values()):
                errs.append(u["symbol"]); continue
            sigs += scan_symbol(u, raw, risk_amt)
            bos += scan_breakouts(u, raw)
            r_, t_ = scan_board(u, raw, risk_amt)
            board += r_
            trades += t_
        except Exception as e:
            errs.append(u["symbol"]); print("ERR", u["symbol"], e)
    sigs.sort(key=lambda s: (-s["score"], s["symbol"]))
    out = {"updated": dt.datetime.now(ad.IST).strftime("%d %b %Y %H:%M IST"), "scanned": len(uni),
           "errors": errs, "diag": DIAG, "breakouts": bos, "board": board, "trades": trades, "date": dt.datetime.now(ad.IST).strftime("%Y-%m-%d"), "capital": capital, "risk_pct": risk_pct, "signals": sigs}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(out, open(OUT, "w"), default=str)
    if len(uni) - len(errs) >= max(1, len(uni) // 2):
        update_history(os.environ.get("GTF_HIST", "docs/history.json"), out["date"], board, trades)
    print(f"{len(uni)} symbols, {len(sigs)} signals, {len(bos)} breakouts, {len(errs)} errors")


if __name__ == "__main__":
    run()
