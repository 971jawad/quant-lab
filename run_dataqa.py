"""DATA QA - is the price data we trade on actually sound?

Two halves.

INTRINSIC checks need no third party. They catch the corruption that an
external cross-check can miss because the reference has the same flaw, and they
run even when every outside source is unreachable:
  * OHLC coherence   high >= max(open,close), low <= min(open,close), high>=low.
                     A violation means the bar was assembled wrongly - a
                     resampling bug, or two sources spliced with mismatched
                     conventions.
  * flat bars        open==high==low==close. A real 24h market never does this;
                     it means a stale print was carried forward as a bar.
  * frozen closes    the same close repeated for days - the signature of a dead
                     feed being forward-filled, which is exactly how WTI sat in
                     the book as a live SHORT for 1,011 days.
  * return outliers  |return| beyond 10 robust sigma, checked against whether
                     the bar is also flat/gapped, so a genuine crash is not
                     confused with a decimal error.
  * weekend bars     Saturday/Sunday bars in an ET-day series indicate the
                     timezone conversion is off by a session.
  * monotone index   strictly increasing, no duplicates.

EXTERNAL cross-checks compare against a source with no code path in common with
ours. Level agreement alone is NOT sufficient - a series can sit at the right
level while being the wrong instrument or the wrong session convention, which
is precisely the WTI case (level looked plausible, return correlation 0.336).
So level and return-correlation are both required.

Stooq was the first choice and is now behind a JavaScript bot-check; FRED times
out from some networks. External checks therefore degrade gracefully: an
unreachable reference is reported as UNVERIFIED, never as a pass.
"""
import io
import json
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from run_shorter import MKT, daily, market_freshness

OUT = Path(__file__).parent / "research"
# Stooq tickers. Chosen to match OUR series' basis wherever possible:
# spot FX and spot gold, cash indices (not futures), so a mismatch is a real
# problem rather than a known and expected futures/spot basis.
STOOQ = {
    "XAUUSD": ("xauusd", "spot gold"),
    "EURUSD": ("eurusd", "spot EURUSD"),
    "USDJPY": ("usdjpy", "spot USDJPY"),
    "MNQ":    ("^ndx",   "Nasdaq-100 cash index"),
    "ES":     ("^spx",   "S&P 500 cash index"),
    "JPXJPY": ("^nkx",   "Nikkei 225 cash index"),
    "WTIUSD": ("cl.f",   "WTI front-month future"),
}
# Where our series and the reference legitimately differ in level, and why.
# A gap inside the band is expected; outside it is a finding.
LEVEL_TOL = {"MNQ": 3.0, "ES": 3.0, "JPXJPY": 3.0,   # CFD vs cash index basis
             "XAUUSD": 1.0, "EURUSD": 0.5, "USDJPY": 0.5, "WTIUSD": 3.0}
CORR_FLOOR = 0.90

# Defects that have been investigated, quantified, and consciously accepted.
# They are reported as KNOWN rather than re-raised as new faults every run, so
# a genuinely new problem still stands out. Nothing is silently suppressed: the
# measured impact is printed with it.
#
# Deliberately NOT auto-repaired. We know what the Nikkei really closed at on
# those days, but hardcoding replacement prices from memory into a price series
# is how fabricated data enters a backtest. Since the measured impact is nil,
# documenting beats patching.
KNOWN_DEFECTS = {
    ("JPXJPY", "2011-03-14"):
        "bad tick: low and close both print exactly 7500 (round number) while the "
        "Nikkei closed ~9620; 2011-03-15 then opens at the bad 7500. IMPACT MEASURED: "
        "leg dev Sharpe 0.38 -> 0.39 if repaired, holdout 0.88 unchanged, only 2 bars "
        "differ (Dec-2011, via the 240d lookback). No effect on holdout or live.",
    ("WTIUSD", "2020-04-27"):
        "-32.2% then +28% during the post-negative-oil dislocation. Plausible for a "
        "CFD in that week but unverifiable now; WTIUSD is excluded from the live book "
        "as stale in any case, so it cannot affect published signals.",
}


def stooq(sym):
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=45).read().decode()
    if "Date" not in raw.split("\n")[0]:
        raise RuntimeError(f"stooq returned no data for {sym}")
    d = pd.read_csv(io.StringIO(raw), parse_dates=["Date"]).set_index("Date")
    d.columns = [c.lower() for c in d.columns]
    return d



def shock_dates(thresh=6.0):
    """Dates on which SEVERAL markets moved hard - i.e. real macro events.

    Needed because "large move that reverses next bar" does not separate a data
    error from a genuine crash: 2020-03-12 (-8.9% MNQ) reversing into 2020-03-13
    (+13.9%) is the COVID crash and its rebound, not a typo. What DOES separate
    them is company: a macro shock moves correlated markets on the same day,
    while a bad print moves exactly one series. So a spike is only treated as a
    fault if no other market moved unusually within a day of it.
    """
    big = {}
    for i in MKT:
        c = daily(i)["close"]
        r = c.pct_change()
        mad = float((r - r.median()).abs().median()) * 1.4826
        if mad <= 0:
            continue
        z = (r - r.median()).abs() / mad
        for t in z[z > thresh].dropna().index:
            big.setdefault(pd.Timestamp(t).normalize(), set()).add(i)
    return {d_: m for d_, m in big.items() if len(m) >= 2}


def intrinsic(inst, d, shocks=None):
    """Checks needing no external reference.

    Tuned to distinguish CORRUPTION from MARKET REALITY. A first cut flagged
    ~860 "weekend" bars per market, WTI's -40.3% on 2020-04-21 and the Nikkei's
    -22.0% on 2011-03-14 -- all of which are correct data. A QA that cries wolf
    on real history gets ignored, so each rule below separates the signature of
    a data fault from the signature of a genuine event.

    Returns (problems, notes): problems are faults, notes are expected-but-worth-
    stating facts about the series.
    """
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    problems, notes = [], []

    bad_hl = int((h < l).sum())
    bad_h = int((h < o.combine(c, max) - 1e-9).sum())
    bad_l = int((l > o.combine(c, min) + 1e-9).sum())
    if bad_hl or bad_h or bad_l:
        problems.append("OHLC incoherent: {} high<low, {} high<max(o,c), {} low>min(o,c)".format(bad_hl, bad_h, bad_l))

    # Saturday bars are impossible on an ET-day 24h series; SUNDAY bars are
    # expected, because the FX/CFD week opens 17:00 ET Sunday and those hours
    # fall in Sunday's calendar day. Only Saturday indicates a timezone fault.
    sat = int((d.index.dayofweek == 5).sum())
    sun = int((d.index.dayofweek == 6).sum())
    if sat:
        problems.append("{} SATURDAY bars - impossible for a 24h market, timezone fault".format(sat))
    if sun:
        notes.append("{} Sunday bars (expected: the week opens 17:00 ET Sunday)".format(sun))

    flat = (o == h) & (h == l) & (l == c)
    nflat = int(flat.sum())
    if nflat:
        dates = ", ".join(str(x.date()) for x in d.index[flat][-3:])
        # a handful on holidays is a thin session, not corruption
        (problems if nflat > 5 else notes).append(
            "{} flat bars o==h==l==c ({})".format(nflat, dates))

    # A frozen close is the dead-feed signature -- this is how WTI stayed in the
    # book as a live SHORT for 1,011 days. Any run of 5+ identical closes in a
    # 24h market is a fault, never a market state.
    grp = (c != c.shift()).cumsum()
    runs = c.groupby(grp).size()
    longest = int(runs.max())
    if longest >= 5:
        end_dt = c.index[grp == runs.idxmax()][-1]
        problems.append("close frozen for {} consecutive bars ending {}".format(longest, end_dt.date()))

    # Outliers: a real crash is large and PERSISTS; a decimal/typo error is
    # large and REVERSES on the very next bar. Only the reversing kind is a
    # data fault. Anything beyond 50% in a day is implausible for these
    # instruments regardless (WTI 2020 excepted, and that one did not reverse).
    r = c.pct_change()
    mad = float((r - r.median()).abs().median()) * 1.4826
    if mad > 0:
        z = (r - r.median()).abs() / mad
        ext = z[z > 10].dropna()
        nxt = r.shift(-1)
        spikes = []
        for t in ext.index:
            a, b = r.loc[t], nxt.loc[t] if t in nxt.index else float("nan")
            if not (b == b and a != 0 and (b / a) < -0.8):
                continue                                   # did not reverse
            day = pd.Timestamp(t).normalize()
            company = any((day + pd.Timedelta(days=k)) in (shocks or {})
                          for k in (-1, 0, 1))
            if company:
                continue                                   # other markets moved too -> real
            spikes.append((t, a))
        known = [x for x in spikes if (inst, str(pd.Timestamp(x[0]).date())) in KNOWN_DEFECTS]
        fresh = [x for x in spikes if x not in known]
        for t, a in known:
            notes.append("KNOWN DEFECT {} {:+.1f}%: {}".format(
                pd.Timestamp(t).date(), a * 100, KNOWN_DEFECTS[(inst, str(pd.Timestamp(t).date()))]))
        if fresh:
            t, a = max(fresh, key=lambda x: abs(x[1]))
            problems.append("NEW spike-and-reverse return (decimal-error signature), worst {:+.1f}% on {}".format(a * 100, pd.Timestamp(t).date()))
        if len(ext):
            t = ext.idxmax()
            notes.append("{} returns beyond 10 robust-sigma, largest {:+.1f}% on {} (persisted - real event)".format(len(ext), r.loc[t] * 100, t.date()))

    if not d.index.is_monotonic_increasing:
        problems.append("index not monotonically increasing")
    dups = int(d.index.duplicated().sum())
    if dups:
        problems.append("{} duplicate timestamps".format(dups))

    return problems, notes


def main():
    print("=" * 78)
    print("DATA QA - our feeds vs Stooq (independent of Yahoo and HistData)")
    print("=" * 78)
    fresh = market_freshness()
    rows, problems = [], []

    print("")
    print("INTRINSIC CHECKS (no external source needed)")
    shocks = shock_dates()
    print("  ({} dates where 2+ markets moved together = real macro events, "
          "excluded from fault detection)".format(len(shocks)))
    intr = {}
    for inst in MKT:
        d = daily(inst)
        probs, notes = intrinsic(inst, d, shocks)
        intr[inst] = {"problems": probs, "notes": notes}
        tag = "{} FAULT(S)".format(len(probs)) if probs else "clean"
        print("  {:9} {:12} {} bars  {} -> {}".format(
            inst, tag, len(d), d.index[0].date(), d.index[-1].date()))
        for x in probs:
            print("              FAULT: " + x)
        for x in notes:
            print("              note : " + x)
        problems += ["{}: {}".format(inst, x) for x in probs]

    print("")
    print("EXTERNAL CROSS-CHECK")
    print(f"{'market':9} {'our last':11} {'our close':>12} {'ref close':>12} "
          f"{'level':>8} {'ret corr':>9} {'stale':>6}  verdict")
    for inst in MKT:
        sym, desc = STOOQ[inst]
        ours = daily(inst)
        try:
            ref = stooq(sym)
        except Exception as e:
            print(f"  {inst:9} reference fetch FAILED: {e}")
            problems.append(f"{inst}: reference unavailable")
            continue
        o_last, r_last = float(ours["close"].iloc[-1]), float(ref["close"].iloc[-1])
        gap = (o_last / r_last - 1) * 100

        a = ours["close"].copy(); a.index = pd.to_datetime(a.index).normalize()
        b = ref["close"].copy();  b.index = pd.to_datetime(b.index).normalize()
        ix = a.index.intersection(b.index)[-500:]
        corr = float(a.reindex(ix).pct_change().corr(b.reindex(ix).pct_change())) \
            if len(ix) > 60 else float("nan")

        tol = LEVEL_TOL[inst]
        bad_level = abs(gap) > tol
        bad_corr = not (corr == corr) or corr < CORR_FLOOR
        bad_stale = fresh[inst] > 10
        verdict = ("STALE" if bad_stale else
                   "WRONG SERIES" if bad_corr else
                   "LEVEL OFF" if bad_level else "ok")
        if verdict != "ok":
            problems.append(f"{inst}: {verdict} (level {gap:+.2f}%, corr {corr:.3f}, "
                            f"{fresh[inst]}d stale)")
        print(f"  {inst:9} {str(ours.index[-1].date()):11} {o_last:>12.4f} "
              f"{r_last:>12.4f} {gap:>+7.2f}% {corr:>9.3f} {fresh[inst]:>5}d  {verdict}")
        rows.append({"market": inst, "reference": desc, "stooq_symbol": sym,
                     "our_last_date": str(ours.index[-1].date()),
                     "our_close": round(o_last, 5), "ref_close": round(r_last, 5),
                     "level_gap_pct": round(gap, 3), "return_corr": round(corr, 4),
                     "days_stale": fresh[inst], "verdict": verdict,
                     "level_tolerance_pct": tol})

    print("\n" + "-" * 78)
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  * {p}")
    else:
        print("every live feed agrees with an independent source on level AND shape")
    json.dump({"checks": rows, "intrinsic": intr, "problems": problems,
               "corr_floor": CORR_FLOOR}, open(OUT / "data_qa.json", "w"), indent=2)
    print("\nwrote research/data_qa.json")


if __name__ == "__main__":
    main()
