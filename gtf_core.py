"""
GTF (Get Together Finance) demand/supply rule engine. Pure logic, no network.

Rules follow the course notes. Where the notes give no number, a default is
set in CFG below (marked ASSUMPTION) so it can be changed in one place.
"""
from dataclasses import dataclass
import math

CFG = {
    "STRONG_MULT": 1.5,      # ASSUMPTION: "strong/explosive" = exciting AND body >= 1.5x avg body of prior 10 bars
    "MAX_BASE": 8,           # bases beyond 5 score 0; >3 bases needs a strong legout (notes)
    "BUF_PCT": 0.05,         # ASSUMPTION: "just above/below" = 0.05% of price, min one tick
    "TICK": 0.05,
    "SIDEWAYS_ATR": 0.4,     # ASSUMPTION: 50 SMA moved < 0.4 ATR over 7 bars = "close to 3 o'clock"
    "MIN_SCORE": 5.0,        # notes: no trade below 5
    "TYPE1_SCORE": 7.0,      # notes: score 7 -> entry type 1, 5-6 -> type 2/3
    "MAX_DIST_PCT": 3.0,     # ASSUMPTION: only alert zones within 3% of price
    "RR": 2.0,               # notes: target 2:1
    "MIN_ZONE_ATR": 0.25,    # zone must be at least 0.25 x ATR(14) thick on its own chart; thinner = noise / bad print
    "LEGOUT_MULT": 1.5,      # legout must be EXPLOSIVE: body >= 1.5 x avg body of the 10 candles before it ...
    "LEGOUT_VS_BASE": 2.0,   # ... and >= 2 x the biggest base candle body (notes: legout explosive, bases boring)
    "LEGIN_MULT": 1.0,       # legin and 2nd exciting candle must be proper: body >= avg body
    "MULTIBASE_LEGOUT": 2.0, # more than 3 bases: legout body >= 2 x avg (notes: strong legout needed)
    "RUN_MAX": 5,            # swing = extreme of the same-colour run (max 5 candles) ending at the legin
    "QUALITY_MIN_CORE": 7,   # only 7/7 zones: fresh(3)+strength(2)+base(2). EMA point is a bonus, not counted. Set 6 to allow 6/7.
}


@dataclass
class Bar:
    t: object
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0


@dataclass
class Zone:
    kind: str            # 'D' or 'S'
    pattern: str         # DBR RBR RBD DBD
    i_legin: int
    i_base0: int
    i_base1: int
    i_legout: int
    prox: float
    distal: float
    nbase: int
    strong: bool
    gap: bool
    two_exc: bool
    t_formed: object = None
    swing: float = 0.0
    legout_x: float = 0.0
    tests: int = 0
    broken: bool = False
    breach_idx: int = -1
    overlap: bool = False
    merged: bool = False
    score: float = 0.0
    parts: dict = None


# ---------------- candles ----------------
def rng(b): return b.h - b.l
def body(b): return abs(b.c - b.o)
def exciting(b): return rng(b) > 0 and body(b) / rng(b) > 0.5     # notes: body > 50% of candle
def color(b): return "G" if b.c > b.o else ("R" if b.c < b.o else "D")
def btop(b): return max(b.o, b.c)
def bbot(b): return min(b.o, b.c)


def clean(bars):
    """Drop zero-range bars (no-trade / closing-price prints). They look like perfect base candles but are not."""
    return [b for b in bars if b.h > b.l]


def too_thin(z, a):
    return abs(z.prox - z.distal) < CFG["MIN_ZONE_ATR"] * a


def avg_body(bars, i, n=10):
    seg = bars[max(0, i - n):i]
    return sum(body(x) for x in seg) / len(seg) if seg else 0.0


def run_extreme(bars, j, demand):
    """Extreme of the same-colour run ending at the legin (origin of the move the legout must close beyond)."""
    col, k, cnt = color(bars[j]), j, 0
    ext = bars[j].h if demand else bars[j].l
    while k >= 0 and color(bars[k]) == col and cnt < CFG["RUN_MAX"]:
        ext = max(ext, bars[k].h) if demand else min(ext, bars[k].l)
        k -= 1
        cnt += 1
    return ext


def real_gap(prev, cur, demand):
    """Notes: only REAL gaps count (market closed, then re-opened) and it must be outside the previous candle's range."""
    try:
        if cur.t.date() == prev.t.date():
            return False                      # inside one session: not a real gap
    except AttributeError:
        pass
    return cur.o > prev.h if demand else cur.o < prev.l


def is_strong(bars, i):
    b = bars[i]
    return exciting(b) and body(b) >= CFG["STRONG_MULT"] * avg_body(bars, i)


# ---------------- zones ----------------
def find_zones(bars):
    """Legin -> base(s) -> legout. Returns zones (unscored)."""
    zones = []
    n = len(bars)
    for i in range(2, n):
        lo = bars[i]
        if not exciting(lo) or color(lo) not in ("G", "R"):
            continue
        demand = color(lo) == "G"
        j = i - 1
        nb = 0
        while j >= 0 and not exciting(bars[j]) and nb < CFG["MAX_BASE"]:
            j -= 1
            nb += 1
        if nb == 0 or j < 0 or not exciting(bars[j]):
            continue
        li = bars[j]
        if color(li) not in ("G", "R"):
            continue
        bases = bars[j + 1:i]
        avg = avg_body(bars, i)
        if avg <= 0:
            continue
        # legin must be a proper exciting candle, legout must be explosive (notes: booster point)
        if body(li) < CFG["LEGIN_MULT"] * avg:
            continue
        max_base = max(body(x) for x in bases)
        if body(lo) < CFG["LEGOUT_MULT"] * avg or body(lo) < CFG["LEGOUT_VS_BASE"] * max_base:
            continue
        if nb > 3 and body(lo) < CFG["MULTIBASE_LEGOUT"] * avg:     # notes: multiple bases need a strong legout
            continue
        # closing concept: legout must CLOSE beyond the whole preceding move (full legin range / origin of the move)
        swing = run_extreme(bars, j, demand)
        if demand and not lo.c > swing:
            continue
        if not demand and not lo.c < swing:
            continue
        strong = True
        reversal = color(li) != color(lo)
        if demand:
            pattern = "DBR" if reversal else "RBR"
            prox = max(btop(x) for x in bases)
            distal = min(x.l for x in bases)
            distal = min(distal, lo.l)                    # exceptional marking: legout wick
            if reversal:
                distal = min(distal, li.l)                # exceptional marking: legin wick (reversal only)
            gap = real_gap(bases[-1], lo, True)
        else:
            pattern = "RBD" if reversal else "DBD"
            prox = min(bbot(x) for x in bases)
            distal = max(x.h for x in bases)
            distal = max(distal, lo.h)
            if reversal:
                distal = max(distal, li.h)
            gap = real_gap(bases[-1], lo, False)
        nx = bars[i + 1] if i + 1 < n else None
        two = bool(nx and exciting(nx) and color(nx) == color(lo) and body(nx) >= CFG["LEGIN_MULT"] * avg
                   and (nx.c > lo.c if demand else nx.c < lo.c))
        zones.append(Zone("D" if demand else "S", pattern, j, j + 1, i - 1, i, prox, distal,
                          nb, strong, gap, two, t_formed=lo.t, swing=swing, legout_x=body(lo) / avg))
    return zones


def update_state(zones, bars):
    """Freshness (tests), breaks, overlap with older same-side zones."""
    n = len(bars)
    for z in zones:
        inside_prev = False
        for k in range(z.i_legout + 1, n):
            b = bars[k]
            if z.kind == "D":
                touch = b.l <= z.prox
                brk = b.c < z.distal
            else:
                touch = b.h >= z.prox
                brk = b.c > z.distal
            if brk:
                z.broken = True
                z.breach_idx = k
                break
            if touch and not inside_prev:
                z.tests += 1
            inside_prev = touch
    # credibility: a zone formed inside an older same-side zone is a reaction of it
    for z in zones:
        for o in zones:
            if o is z or o.kind != z.kind or o.i_legout >= z.i_legin:
                continue
            lo_, hi_ = min(o.prox, o.distal), max(o.prox, o.distal)
            zl, zh = min(z.prox, z.distal), max(z.prox, z.distal)
            if zl <= hi_ and zh >= lo_ and not (o.broken and o.breach_idx < z.i_base0):
                z.overlap = True
    return zones


def merge_lotl(zones):
    """LOTL: level on top of a level (stacked right after the first legout) -> one trade,
    entry at the upper zone's proximal, stop beyond the lower zone's distal."""
    for z in zones:
        for o in zones:
            if o is z or o.kind != z.kind:
                continue
            if z.kind == "D" and o.i_base0 > z.i_legout and o.i_base0 <= z.i_legout + 2 and o.prox > z.prox:
                o.merged, o.lotl_distal, o.lotl_with = True, z.distal, z
            if z.kind == "S" and o.i_base0 > z.i_legout and o.i_base0 <= z.i_legout + 2 and o.prox < z.prox:
                o.merged, o.lotl_distal, o.lotl_with = True, z.distal, z
    return zones


def ema(vals, n):
    if len(vals) < n:
        return []
    k = 2 / (n + 1)
    out = [sum(vals[:n]) / n]
    for v in vals[n:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def score_zones(zones, bars):
    closes = [b.c for b in bars]
    e20, e50 = ema(closes, 20), ema(closes, 50)
    for z in zones:
        fresh = 3.0 if z.tests == 0 else (1.5 if z.tests == 1 else 0.0)
        strength = 2.0 if (z.gap or z.two_exc) else 1.0
        base = 2.0 if z.nbase <= 3 else (1.0 if z.nbase <= 5 else 0.0)
        lo_, hi_ = min(z.prox, z.distal), max(z.prox, z.distal)
        e_pts = 0.0
        for e in (e20, e50):
            if e and lo_ <= e[-1] <= hi_:
                e_pts = 1.0
        z.parts = {"fresh": fresh, "strength": strength, "base": base, "ema": e_pts}
        z.core = fresh + strength + base
        z.score = z.core + e_pts
    return zones


def build_zones(bars):
    zs = find_zones(bars)
    update_state(zs, bars)
    merge_lotl(zs)
    score_zones(zs, bars)
    return zs


def valid(z):
    return not z.broken


def last_breach(zones):
    ev = [(z.breach_idx, z.kind) for z in zones if z.broken and z.breach_idx >= 0]
    return max(ev)[1] if ev else None


# ---------------- trend (50 SMA "clock") ----------------
def sma(vals, n):
    return [sum(vals[i - n + 1:i + 1]) / n for i in range(n - 1, len(vals))]


def atr(bars, n=14):
    if len(bars) < n + 1:
        return 0.0
    trs = [max(bars[i].h - bars[i].l, abs(bars[i].h - bars[i - 1].c), abs(bars[i].l - bars[i - 1].c))
           for i in range(1, len(bars))]
    return sum(trs[-n:]) / n


def trend_clock(bars):
    closes = [b.c for b in bars]
    s = sma(closes, 50)
    if len(s) < 9:
        return "SIDEWAYS"
    a = atr(bars) or 1e-9
    d = (s[-1] - s[-8]) / a          # 7 candles back, per notes
    rising = s[-1] > s[-2]
    if abs(d) < CFG["SIDEWAYS_ATR"]:
        return "SIDEWAYS"
    if d > 0 and rising:
        return "UP"
    if d < 0 and not rising:
        return "DOWN"
    return "SIDEWAYS"


def trend_advanced(zones):
    """Two same-side zone breaches in a row decide trend; one breach = sideways."""
    ev = sorted([(z.breach_idx, z.kind) for z in zones if z.broken and z.breach_idx >= 0])
    if not ev:
        return "SIDEWAYS"
    last = ev[-1][1]
    run = 0
    for _, k in reversed(ev):
        if k == last:
            run += 1
        else:
            break
    if run >= 2:
        return "UP" if last == "S" else "DOWN"      # supply breached twice -> up; demand twice -> down
    return "SIDEWAYS"


# ---------------- curve / location (HTF) ----------------
def location(htf_zones, cmp_):
    sup = [z for z in htf_zones if z.kind == "S" and valid(z) and z.tests == 0 and z.distal > cmp_]
    dem = [z for z in htf_zones if z.kind == "D" and valid(z) and z.tests == 0 and z.distal < cmp_]
    if not sup or not dem:
        return "EQ", None, None          # notes: missing fresh side -> treat as equilibrium
    s = min(sup, key=lambda z: z.prox)
    d = max(dem, key=lambda z: z.prox)
    if cmp_ >= s.prox:
        return "VERY_HIGH", s, d
    if cmp_ <= d.prox:
        return "VERY_LOW", s, d
    third = (s.prox - d.prox) / 3.0
    if cmp_ >= s.prox - third:
        return "HIGH", s, d
    if cmp_ <= d.prox + third:
        return "LOW", s, d
    return "EQ", s, d


def allowed_side(loc, itf_trend):
    if loc in ("HIGH", "VERY_HIGH"):
        return "SELL"
    if loc in ("LOW", "VERY_LOW"):
        return "BUY"
    return {"UP": "BUY", "DOWN": "SELL"}.get(itf_trend)      # sideways -> ignore


# ---------------- trade plan ----------------
def rnd(x):
    t = CFG["TICK"]
    return round(round(x / t) * t, 2)


def plan(z):
    buf = max(CFG["TICK"], z.prox * CFG["BUF_PCT"] / 100.0)
    distal = getattr(z, "lotl_distal", z.distal) if z.merged else z.distal
    if z.kind == "D":
        entry, sl = z.prox + buf, distal - buf
        side = "BUY"
        tgt = entry + CFG["RR"] * (entry - sl)
    else:
        entry, sl = z.prox - buf, distal + buf
        side = "SELL"
        tgt = entry - CFG["RR"] * (sl - entry)
    return {"side": side, "entry": rnd(entry), "sl": rnd(sl), "target": rnd(tgt), "risk": rnd(abs(entry - sl))}


def target_permitted(p, others):
    """Notes: target must be permitted by the setup -> no opposing valid zone between entry and target."""
    for o in others:
        if not valid(o):
            continue
        if p["side"] == "BUY" and o.kind == "S" and p["entry"] < o.prox < p["target"]:
            return False
        if p["side"] == "SELL" and o.kind == "D" and p["target"] < o.prox < p["entry"]:
            return False
    return True


def lots_for(p, lot, risk_amt):
    per_lot = p["risk"] * lot
    return int(risk_amt // per_lot) if per_lot > 0 else 0


def in_zone(z, px):
    lo_, hi_ = min(z.prox, z.distal), max(z.prox, z.distal)
    return lo_ <= px <= hi_


def entry_status(z, bars, cmp_):
    """Type 1 (score>=7): set-and-forget limit. Type 2/3 (5-6): needs the two-candle confirmation
    on closed execution bars. Returns (entry_type, status)."""
    p = plan(z)
    dist = abs(cmp_ - p["entry"]) / cmp_ * 100
    if z.score >= CFG["TYPE1_SCORE"]:
        correct_side = cmp_ >= p["entry"] if z.kind == "D" else cmp_ <= p["entry"]
        if correct_side and dist <= CFG["MAX_DIST_PCT"]:
            return 1, "SET_LIMIT"
        return 1, None
    if len(bars) < 2:
        return 2, None
    a, b = bars[-2], bars[-1]
    if not in_zone(z, a.c):
        return 2, None
    if in_zone(z, b.o):
        return 2, "TYPE2_TRIGGER"
    left = b.c > z.prox if z.kind == "D" else b.c < z.prox
    if left and not in_zone(z, b.o):
        risk = p["risk"]
        chased = (cmp_ - p["entry"] > 0.5 * risk) if z.kind == "D" else (p["entry"] - cmp_ > 0.5 * risk)
        return 3, None if chased else "TYPE3_CONFIRMED"
    return 2, None


# ---------------- full evaluation ----------------
def evaluate(symbol, ltf, itf, htf, cmp_, lot, risk_amt):
    """Returns list of signal dicts for one symbol."""
    ltf, itf, htf = clean(ltf), clean(itf), clean(htf)
    a_ltf = atr(ltf)
    lz = build_zones(ltf)
    iz = build_zones(itf)
    hz = build_zones(htf)
    t_itf = trend_clock(itf)
    adv = trend_advanced(iz)
    loc, s_z, d_z = location(hz, cmp_)
    side_ok = allowed_side(loc, t_itf)
    out = []
    if side_ok is None:
        return out
    others = [z for z in lz + iz if valid(z)]
    for z in lz:
        if not valid(z) or z.score < CFG["MIN_SCORE"] or z.core < CFG["QUALITY_MIN_CORE"]:
            continue
        if too_thin(z, a_ltf):
            continue
        if ("BUY" if z.kind == "D" else "SELL") != side_ok:
            continue
        if z.overlap and not z.strong:            # reaction zone: only tradable with a good (strong) closing
            continue
        # notes: buying in a downtrend / selling in an uptrend is dangerous. Only allowed after the ITF
        # has breached an opposite zone (supply for buys, demand for sells) as the MOST RECENT breach.
        against = (side_ok == "BUY" and t_itf == "DOWN") or (side_ok == "SELL" and t_itf == "UP")
        if against and last_breach(iz) != ("S" if side_ok == "BUY" else "D"):
            continue
        p = plan(z)
        if not target_permitted(p, [o for o in others if o is not z]):
            continue
        et, st = entry_status(z, ltf, cmp_)
        if not st:
            ok_side = cmp_ >= p["entry"] if z.kind == "D" else cmp_ <= p["entry"]
            if CFG["QUALITY_MIN_CORE"] < 7 and ok_side and abs(cmp_ - p["entry"]) / cmp_ * 100 <= CFG["MAX_DIST_PCT"]:
                st = "WATCH"        # valid zone nearby, entry condition not met yet
            else:
                continue
        lots = lots_for(p, lot, risk_amt)
        out.append({
            "symbol": symbol, "status": st, "entry_type": et, "score": z.score, "parts": z.parts,
            "pattern": z.pattern, "zone": (rnd(z.distal), rnd(z.prox)), "tests": z.tests,
            "lotl": z.merged, "t_zone": str(z.t_formed), "loc": loc, "itf_trend": t_itf,
            "plan": p, "lots": lots, "lot": lot, "cmp": rnd(cmp_),
            "swing": rnd(z.swing), "legout_x": round(z.legout_x, 1), "t_legin": str(ltf[z.i_legin].t), "width": rnd(abs(z.prox - z.distal)),
            "risk_pct": round(p["risk"] / p["entry"] * 100, 2),
        })
    return out
