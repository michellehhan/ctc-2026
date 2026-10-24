#!/usr/bin/env python3
"""Stage 1 research: fit settlement models on the 600 research seasons and write model.json.

    python3 research.py            # prints all tables, writes model.json

Findings that shape the model (see the printed tables for the numbers):
  * ISLE and AQUA settle on an exact line in cumulative final damage_pct.
  * PORT, GRID, FERRY have an extra "outage" term: how long the storm sat over their sites
    (seconds the storm radius covered the site, weighted by site weight and intensity). That term is
    observable live from the `storm` observations, so the bot tracks it. Without it the linear fit is
    still fine (resid ~0.8-1.7); with it resid ~0.3-0.5.
  * Season insured loss is (almost exactly) 9*PORT + 6*ISLE + 2.5*AQUA + 7*GRID + 1.2*FERRY cumulative
    damage_pct, so a damage report pins down the loss too.
  * MAT = a + b*sqrt(season loss). RECON = a + b*sqrt(dollar-weighted PORT/GRID/FERRY damage).
    GEN = a + b*sqrt(GRID damage) + c*GRID outage exposure.
  * HAVEN = 75 - 0.06*clip(loss-400, 0, 900): retention 400, layer 400..1300.
    REIN  = 90 - 0.035*clip(loss-1300, 0, ~470): attaches at 1300. Top of REIN layer is beyond the data.
Only numpy + stdlib.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "starter_kit"))
from history import Season, load  # noqa: E402

ISLAND = ["PORT", "ISLE", "AQUA", "GRID", "FERRY"]
REBUILD = ["RECON", "MAT", "GEN"]
INSURER = ["HAVEN", "REIN"]
STOCKS = ISLAND + REBUILD + INSURER
ALL = STOCKS + ["ARCH", "REBLD"]
OBS_DT = 2.0  # real seconds between storm observations


def lstsq(cols, y):
    X = np.column_stack([np.ones(len(y))] + list(cols))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, r


def exposure_of(season, case):
    """Outage exposure per island company: sum over storm observations and sites of
    OBS_DT * weight * intensity when the storm radius covers the site. Also the unweighted variant."""
    obs = season.messages(type="storm")
    if not obs:
        return {k: 0.0 for k in ISLAND}, {k: 0.0 for k in ISLAND}
    ox = np.array([m["x"] for m in obs]); oy = np.array([m["y"] for m in obs])
    orad = np.array([m["radius"] for m in obs]); oint = np.array([m.get("intensity", 1.0) for m in obs])
    expo_i, expo = {}, {}
    for k in ISLAND:
        ti = t = 0.0
        for site in case["companies"][k]["sites"]:
            inside = np.hypot(ox - site["x"], oy - site["y"]) < orad
            t += OBS_DT * site["weight"] * inside.sum()
            ti += OBS_DT * site["weight"] * (oint * inside).sum()
        expo_i[k], expo[k] = float(ti), float(t)
    return expo_i, expo


def main():
    data = load(HERE / "history")
    case = json.load(open(HERE / "case.json"))
    S = data.seasons
    n = len(S)
    print(f"loaded {n} seasons")
    model = {"n_seasons": n, "etfs": {k: v["basket"] for k, v in case["etfs"].items()}, "limits": case["limits"]}

    cum = {k: np.array([sum(st["damage_pct"][k] for st in s.storms) for s in S]) for k in ISLAND}
    loss = np.array([s.season_loss_musd for s in S])
    sett = {k: np.array([s.settlement[k] for s in S]) for k in ALL}
    print("computing storm exposure per season...", flush=True)
    ex = [exposure_of(s, case) for s in S]
    expo_i = {k: np.array([e[0][k] for e in ex]) for k in ISLAND}
    expo = {k: np.array([e[1][k] for e in ex]) for k in ISLAND}

    # ---------------------------------------------------------------- 2. island businesses
    print("\n== 2. Island businesses: settlement vs cumulative final damage_pct ==")
    print(f"{'sym':6} {'base':>8} {'slope':>8} {'lin resid':>10} {'quad resid':>10} | {'+expo coef':>10} {'resid':>7}  (expo = intensity-weighted seconds storm over sites)")
    model["islands"] = {}
    for k in ISLAND:
        y = sett[k]
        b1, r1 = lstsq([cum[k]], y)
        b2, r2 = lstsq([cum[k], cum[k] ** 2], y)
        bi, ri = lstsq([cum[k], expo_i[k]], y)
        bu, ru = lstsq([cum[k], expo[k]], y)
        if ru.std() < ri.std():
            bi, ri, kind = bu, ru, "expo"
        else:
            kind = "expo_i"
        print(f"{k:6} {b1[0]:8.3f} {b1[1]:8.4f} {r1.std():10.3f} {r2.std():10.3f} | {bi[2]:10.4f} {ri.std():7.3f}  [{kind}] base_expo={bi[0]:.3f} slope_expo={bi[1]:.4f}")
        model["islands"][k] = {
            "base": float(b1[0]), "slope": float(b1[1]), "resid_std": float(r1.std()),
            "expo_kind": kind, "base_expo": float(bi[0]), "slope_expo": float(bi[1]), "expo_coef": float(bi[2]),
            "resid_std_expo": float(ri.std()),
            "mean_expo_per_dmg": float(np.sum(expo_i[k] if kind == "expo_i" else expo[k]) / max(np.sum(cum[k]), 1e-9)),
        }
    print("Quadratic adds nothing; linear is the right shape. ISLE/AQUA exact. PORT/GRID/FERRY improve a lot with the exposure term.")

    # ---------------------------------------------------------------- loss from damage
    bl, rl = lstsq([cum[k] for k in ISLAND], loss)
    print(f"\n== Season loss ($M) from cumulative damage: loss = {bl[0]:.2f} + " +
          " + ".join(f"{c:.3f}*{k}" for c, k in zip(bl[1:], ISLAND)) + f"   resid std {rl.std():.3f}")
    model["loss_from_damage"] = {"intercept": float(bl[0]), **{k: float(c) for k, c in zip(ISLAND, bl[1:])}, "resid_std": float(rl.std())}

    # ---------------------------------------------------------------- 3. rebuilders
    print("\n== 3. Rebuilders ==")
    model["rebuilders"] = {}
    base_val = {k: model["islands"][k]["base"] for k in ISLAND}
    # RECON: sqrt of dollar-weighted damage of PORT+GRID+FERRY (weights = undamaged share value)
    wrecon = {k: base_val[k] for k in ("PORT", "GRID", "FERRY")}
    xr = sum(cum[k] * wrecon[k] for k in wrecon)
    for label, cols in [("linear dmg", [xr]), ("sqrt dmg", [np.sqrt(xr)]), ("log1p dmg", [np.log1p(xr)]),
                        ("sqrt unweighted", [np.sqrt(cum["PORT"] + cum["GRID"] + cum["FERRY"])])]:
        b, r = lstsq(cols, sett["RECON"])
        print(f"RECON {label:16} coef {np.round(b, 4)} resid {r.std():.3f}")
    b, r = lstsq([np.sqrt(xr)], sett["RECON"])
    model["rebuilders"]["RECON"] = {"form": "a+b*sqrt(sum w_k*dmg_k)", "weights": wrecon, "a": float(b[0]), "b": float(b[1]), "resid_std": float(r.std())}
    # MAT: sqrt of season loss (all repairs)
    xm_d = sum(cum[k] * base_val[k] for k in ISLAND)
    for label, cols in [("linear loss", [loss]), ("sqrt loss", [np.sqrt(loss)]), ("log1p loss", [np.log1p(loss)]), ("sqrt $dmg", [np.sqrt(xm_d)])]:
        b, r = lstsq(cols, sett["MAT"])
        print(f"MAT   {label:16} coef {np.round(b, 4)} resid {r.std():.3f}")
    b, r = lstsq([np.sqrt(loss)], sett["MAT"])
    model["rebuilders"]["MAT"] = {"form": "a+b*sqrt(season_loss)", "a": float(b[0]), "b": float(b[1]), "resid_std": float(r.std())}
    # GEN: sqrt GRID damage + GRID outage exposure
    for label, cols in [("linear GRID", [cum["GRID"]]), ("sqrt GRID", [np.sqrt(cum["GRID"])]), ("log1p GRID", [np.log1p(cum["GRID"])]),
                        ("sqrt GRID+expo", [np.sqrt(cum["GRID"]), expo_i["GRID"]])]:
        b, r = lstsq(cols, sett["GEN"])
        print(f"GEN   {label:16} coef {np.round(b, 4)} resid {r.std():.3f}")
    b0, r0 = lstsq([np.sqrt(cum["GRID"])], sett["GEN"])
    b, r = lstsq([np.sqrt(cum["GRID"]), expo_i["GRID"]], sett["GEN"])
    model["rebuilders"]["GEN"] = {"form": "a+b*sqrt(GRID dmg) [+c*GRID expo_i]", "a": float(b0[0]), "b": float(b0[1]), "resid_std": float(r0.std()),
                                  "a_expo": float(b[0]), "b_expo": float(b[1]), "c_expo": float(b[2]), "resid_std_expo": float(r.std()),
                                  "mean_expo_per_dmg": model["islands"]["GRID"]["mean_expo_per_dmg"] if model["islands"]["GRID"]["expo_kind"] == "expo_i"
                                  else float(np.sum(expo_i["GRID"]) / max(np.sum(cum["GRID"]), 1e-9))}

    # ---------------------------------------------------------------- 4. insurers
    print("\n== 4. Insurers: piecewise-linear layer fit on season loss ($M) ==")
    model["insurers"] = {}
    for sym in INSURER:
        y = sett[sym]; top = float(y.max()); best = None
        for a in np.arange(50, 1600, 5):
            x0 = np.clip(loss - a, 0, None)
            for L in np.arange(50, 1600, 10):
                x = np.minimum(x0, L)
                b, r = lstsq([x], y)
                if best is None or r.std() < best[0]:
                    best = (r.std(), float(a), float(L), b)
        rs, att, lay, b = best
        covered = loss.max() >= att + lay
        print(f"{sym}: top {b[0]:.2f}  attaches at {att:.0f}  layer width {lay:.0f} (-> {att + lay:.0f}{'' if covered else ', upper bound NOT observed in data'})"
              f"  slope {-b[1]:.4f} $/share per $M  resid {rs:.4f}  min settle {y.min():.2f}  max loss seen {loss.max():.0f}")
        model["insurers"][sym] = {"top": float(b[0]), "attach": att, "layer": lay, "slope": float(-b[1]), "resid_std": float(rs),
                                  "layer_top_observed": bool(covered)}
    print("-> below 400 nobody pays; HAVEN 400..1300; REIN from 1300 up. Loss percentiles 10/50/90:",
          np.round(np.percentile(loss, [10, 50, 90]), 0))

    # ---------------------------------------------------------------- 5. prelim vs final
    print("\n== 5. Preliminary vs final damage report: final = a + b*prelim ==")
    print(f"{'field':8} {'a':>7} {'b':>7} {'bias(final-prelim)':>20} {'resid std':>10} {'resid std (final-prelim)':>24} {'n':>5}")
    model["prelim"] = {}
    pairs = defaultdict(list)
    for s in S:
        for rnd in (1, 2, 3):
            reps = s.messages(rnd, type="damage_report")
            p = next((m for m in reps if m["stage"] == "preliminary"), None)
            f = next((m for m in reps if m["stage"] == "final"), None)
            if p and f:
                for k in ISLAND:
                    pairs[k].append((p["damage_pct"][k], f["damage_pct"][k]))
                pairs["loss"].append((p["industry_loss_musd"], f["industry_loss_musd"]))
    for k, pf in pairs.items():
        pf = np.array(pf); x, y = pf[:, 0], pf[:, 1]
        b, r = lstsq([x], y)
        d = y - x
        print(f"{k:8} {b[0]:7.3f} {b[1]:7.4f} {d.mean():20.3f} {r.std():10.3f} {d.std():24.3f} {len(x):5d}")
        model["prelim"][k] = {"a": float(b[0]), "b": float(b[1]), "bias": float(d.mean()), "resid_std": float(r.std()), "diff_std": float(d.std())}

    # ---------------------------------------------------------------- 6. advisory accuracy
    print("\n== 6. Official advisory accuracy by advisory number (vs the storm's steering pattern; 'eff' treats the pattern as 'erratic' once the storm has turned erratic) ==")
    model["advisory"] = {"official": {}, "wire": {}}
    off = defaultdict(lambda: Counter()); wire = defaultdict(lambda: Counter())
    port_dmg_given_direct = defaultdict(list); port_dmg_all = []
    for s in S:
        for st in s.storms:
            rnd = st["round"]; true = st["pattern"]; erratic_at = st["turned_erratic_at"]
            port_dmg_all.append(st["damage_pct"]["PORT"])
            for m in s.messages(rnd, type="advisory"):
                eff = "erratic" if (erratic_at is not None and m["age"] >= erratic_at) else true
                key = m["number"] if m["source"] == "official" else min(8, max(1, int((m["age"] - 30) // 60) + 1))
                c = off[key] if m["source"] == "official" else wire[key]
                c["n"] += 1
                c["ok"] += m["outlook"] == true
                c["ok_eff"] += m["outlook"] == eff
                if m["outlook"] == "direct":
                    c["direct_calls"] += 1
                    c["direct_ok"] += true == "direct"
                    c["direct_port_dmg"] += st["damage_pct"]["PORT"]
                    if m["source"] == "official":
                        port_dmg_given_direct[key].append(st["damage_pct"]["PORT"])
                if true == "direct":
                    c["true_direct"] += 1
                    c["direct_recall"] += m["outlook"] == "direct"
    for name, table in (("official", off), ("wire", wire)):
        print(f"\n{name}:  {'#':>2} {'n':>5} {'acc':>6} {'acc_eff':>8} | {'direct calls':>12} {'precision':>9} {'recall':>7} {'mean PORT dmg|direct call':>26}")
        for key in sorted(table):
            c = table[key]
            prec = c["direct_ok"] / c["direct_calls"] if c["direct_calls"] else float("nan")
            rec = c["direct_recall"] / c["true_direct"] if c["true_direct"] else float("nan")
            mp = c["direct_port_dmg"] / c["direct_calls"] if c["direct_calls"] else float("nan")
            print(f"           {key:>2} {c['n']:5d} {c['ok'] / c['n']:6.2f} {c['ok_eff'] / c['n']:8.2f} | {c['direct_calls']:12d} {prec:9.2f} {rec:7.2f} {mp:26.1f}")
            model["advisory"][name][str(key)] = {"n": c["n"], "acc": c["ok"] / c["n"], "acc_eff": c["ok_eff"] / c["n"],
                                                 "direct_calls": c["direct_calls"], "direct_precision": prec if c["direct_calls"] else None,
                                                 "direct_recall": rec if c["true_direct"] else None,
                                                 "mean_port_dmg_given_direct": mp if c["direct_calls"] else None}
    print(f"mean PORT damage per storm overall: {np.mean(port_dmg_all):.1f}  (share of storms with PORT dmg > 20: {np.mean(np.array(port_dmg_all) > 20):.2f})")
    pat = Counter(st["pattern"] for s in S for st in s.storms)
    print("pattern frequencies:", dict(pat))
    model["advisory"]["pattern_freq"] = {k: v / sum(pat.values()) for k, v in pat.items()}

    # ---------------------------------------------------------------- 7. house behaviour
    print("\n== 7. House price at the FINAL damage report vs the 'truth' (settlement for round 3, next-round open for rounds 1-2) ==")
    print("gap = truth - house price (positive: house too low). Also the gap 60 real seconds after the report.")
    model["house"] = {}
    gaps = defaultdict(lambda: defaultdict(list))
    for s in S:
        times = s.times; rounds = s.rounds
        for rnd in (1, 2, 3):
            f = next((m for m in s.messages(rnd, type="damage_report") if m["stage"] == "final"), None)
            if not f:
                continue
            idx = next((i for i, t in enumerate(times) if rounds[i] == rnd and t >= f["time"]), None)
            if idx is None:
                continue
            idx2 = min(idx + 30, len(times) - 1)  # 30 samples * 2 s = 60 real seconds later
            if rnd == 3:
                truth = {k: s.settlement[k] for k in ALL}
            else:
                j = next(i for i, r in enumerate(rounds) if r == rnd + 1)
                truth = {k: s.prices[k][j] for k in ALL}
            for k in ALL:
                gaps[k][f"r{rnd}_at"].append(truth[k] - s.prices[k][idx])
                gaps[k][f"r{rnd}_60s"].append(truth[k] - s.prices[k][idx2])
    print(f"{'sym':6} " + " ".join(f"{f'r{r} gap@rep':>14} {f'r{r} gap+60s':>14}" for r in (1, 2, 3)))
    for k in ALL:
        row = []
        model["house"][k] = {}
        for rnd in (1, 2, 3):
            a = np.array(gaps[k][f"r{rnd}_at"]); b = np.array(gaps[k][f"r{rnd}_60s"])
            row.append(f"{a.mean():6.2f}±{a.std():5.2f} {b.mean():6.2f}±{b.std():5.2f}")
            model["house"][k][f"r{rnd}"] = {"gap_at_report_mean": float(a.mean()), "gap_at_report_std": float(a.std()),
                                             "gap_60s_mean": float(b.mean()), "gap_60s_std": float(b.std())}
        print(f"{k:6} " + " ".join(f"{x:>29}" for x in row))


    # ---------------------------------------------------------------- per-storm table for Monte-Carlo in valuation.py
    # Each historical storm: final damage_pct per island company, its insured loss, and its exposure term.
    # valuation.py draws from these to price the storms that have not happened yet (rounds 1-2).
    # "damage_raw" = damage as a share of the value that was still standing before the storm: a storm damages only
    # what is left (cumulative damage never exceeds 100 and raw damage is uncorrelated with prior damage), so the
    # MC applies raw damage to the remaining value.
    storms_tbl = {"damage": {k: [] for k in ISLAND}, "damage_raw": {k: [] for k in ISLAND}, "loss": [], "expo": {k: [] for k in ISLAND}}
    for s in S:
        before = {k: 0.0 for k in ISLAND}
        for rnd, st in enumerate(s.storms):
            sub = Season(s.number, s.folder, [st], 0.0)
            sub.__dict__["feed"] = [m for m in s.feed if m["round"] == rnd + 1]
            ei, eu = exposure_of(sub, case)
            for k in ISLAND:
                d = st["damage_pct"][k]
                storms_tbl["damage"][k].append(d)
                storms_tbl["damage_raw"][k].append(min(100.0, d / max(1.0 - before[k] / 100.0, 1e-6)))
                storms_tbl["expo"][k].append(ei[k] if model["islands"][k]["expo_kind"] == "expo_i" else eu[k])
                before[k] += d
            storms_tbl["loss"].append(st["industry_loss_musd"])
    model["storms"] = storms_tbl
    model["open_price_mean"] = {k: float(np.mean([s.prices[k][0] for s in S])) for k in ALL}
    model["open_price_std"] = {k: float(np.std([s.prices[k][0] for s in S])) for k in ALL}
    print("\nper-storm table:", len(storms_tbl["loss"]), "storms; round-1 open means:",
          {k: round(v, 2) for k, v in model["open_price_mean"].items()})

    # ---------------------------------------------------------------- 8. save
    out = HERE / "model.json"
    json.dump(model, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
