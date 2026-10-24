#!/usr/bin/env python3
"""Stage 2: fair values from model.json.

    from valuation import fair_values, fair_and_sigma
    fairs = fair_values(cumulative_damage_pct, season_loss_musd)                      # settlement if nothing else happens
    fairs = fair_values(cum, loss, storms_remaining=2, exposure=expo, prelim=p, prelim_loss=L)

Settlement = f(cumulative final damage_pct, season loss, storm exposure). Before the last final report the
settlement is unknown, so the fair value is the EXPECTED settlement: a Monte-Carlo over the 1800 historical
storms for the storms that have not happened yet (`storms_remaining`), and the bias-corrected preliminary
report for the storm whose final report is still to come (`prelim`). `exposure` is the outage term
(seconds the storm covered each company's sites, weighted; see research.py); when None it is estimated from
the damage. Pure functions, numpy only; one call is ~1 ms.

Validation (python3 valuation.py): compares the fair at the round-1/2/3 opens with the house price there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = json.load(open(HERE / "model.json"))

ISLAND = ["PORT", "ISLE", "AQUA", "GRID", "FERRY"]
STOCKS = ISLAND + ["RECON", "MAT", "GEN", "HAVEN", "REIN"]
ETFS = MODEL["etfs"]
SYMBOLS = STOCKS + list(ETFS)
LIMITS = MODEL["limits"]
_N_SCEN = 1500
_rng = np.random.default_rng(12345)

_STORM_DMG = {k: np.array(MODEL["storms"]["damage_raw"][k]) for k in ISLAND}  # share of REMAINING value
_LOSS_COEF = {k: MODEL["loss_from_damage"][k] for k in ISLAND}
_STORM_EXPO = {k: np.array(MODEL["storms"]["expo"][k]) for k in ISLAND}
_STORM_LOSS = np.array(MODEL["storms"]["loss"])
_N_STORMS = len(_STORM_LOSS)
# pre-drawn scenario indices: remaining=1 -> every storm once; remaining=2,3 -> random combinations
_SCEN = {1: np.arange(_N_STORMS)[:, None]}
for _r in (2, 3):
    _SCEN[_r] = _rng.integers(0, _N_STORMS, size=(_N_SCEN, _r))


def _price_arrays(cum: dict, loss: np.ndarray, expo: dict) -> dict:
    """Vectorised settlement model. cum/expo: dict sym -> array; loss: array. Returns dict sym -> array."""
    out = {}
    for k in ISLAND:
        m = MODEL["islands"][k]
        out[k] = m["base_expo"] + m["slope_expo"] * cum[k] + m["expo_coef"] * expo[k]
    r = MODEL["rebuilders"]["RECON"]
    out["RECON"] = r["a"] + r["b"] * np.sqrt(np.maximum(sum(w * cum[k] for k, w in r["weights"].items()), 0.0))
    mt = MODEL["rebuilders"]["MAT"]
    out["MAT"] = mt["a"] + mt["b"] * np.sqrt(np.maximum(loss, 0.0))
    g = MODEL["rebuilders"]["GEN"]
    out["GEN"] = g["a_expo"] + g["b_expo"] * np.sqrt(np.maximum(cum["GRID"], 0.0)) + g["c_expo"] * expo["GRID"]
    for k in ("HAVEN", "REIN"):
        i = MODEL["insurers"][k]
        out[k] = i["top"] - i["slope"] * np.clip(loss - i["attach"], 0.0, i["layer"])
    for etf, basket in ETFS.items():
        out[etf] = sum(w * out[k] for k, w in basket.items())
    return out


def _known_state(cum, loss, exposure, prelim, prelim_loss, prelim_is_estimate=False):
    """Fold the prelim (bias-corrected unless it is our own track estimate) into the known damage;
    estimate exposure when not given."""
    c = {k: float(cum.get(k, 0.0)) for k in ISLAND}
    L = float(loss)
    if prelim:
        bias = {k: (0.0 if prelim_is_estimate else MODEL["prelim"][k]["bias"]) for k in ISLAND}
        lbias = 0.0 if prelim_is_estimate else MODEL["prelim"]["loss"]["bias"]
        for k in ISLAND:
            c[k] += max(0.0, float(prelim.get(k, 0.0)) + bias[k])
        if prelim_loss is not None:
            L += max(0.0, float(prelim_loss) + lbias)
        else:  # derive the loss from the damage (exact linear relation found in research)
            lf = MODEL["loss_from_damage"]
            L += max(0.0, sum(lf[k] * max(0.0, float(prelim.get(k, 0.0)) + bias[k]) for k in ISLAND))
    if exposure is None:
        e = {k: MODEL["islands"][k]["mean_expo_per_dmg"] * c[k] for k in ISLAND}
    else:
        e = {k: float(exposure.get(k, 0.0)) for k in ISLAND}
        if prelim:  # exposure passed in covers only reported storms? caller passes the running total, so nothing to add
            pass
    return c, L, e


def _future(c: dict, L: float, e: dict, r: int):
    """Apply r drawn storms in sequence; each damages the value still standing. Returns (cum, loss, expo) arrays."""
    idx = _SCEN[r]
    cum = {k: np.full(idx.shape[0], c[k]) for k in ISLAND}
    expo = {k: np.full(idx.shape[0], e[k]) for k in ISLAND}
    loss = np.full(idx.shape[0], L)
    for j in range(r):
        for k in ISLAND:
            d = _STORM_DMG[k][idx[:, j]] * np.maximum(1.0 - cum[k] / 100.0, 0.0)
            cum[k] = cum[k] + d
            loss = loss + _LOSS_COEF[k] * d
            expo[k] = expo[k] + _STORM_EXPO[k][idx[:, j]]
    return cum, loss, expo


def fair_values(cumulative_damage_pct: dict, season_loss_musd: float, storms_remaining: int = 0,
                exposure: dict | None = None, prelim: dict | None = None, prelim_loss: float | None = None,
                prelim_is_estimate: bool = False) -> dict:
    """Expected settlement per symbol (stocks and ETFs).

    cumulative_damage_pct: final damage_pct per island company, summed over storms with a FINAL report.
    season_loss_musd: insured loss summed over storms with a FINAL report.
    storms_remaining: storms with no damage report yet (includes the current storm until its prelim arrives).
    exposure: running total of the outage term per island company (from `storm` observations), or None.
    prelim/prelim_loss: the preliminary report of the storm whose final is still pending, or None.
    """
    c, L, e = _known_state(cumulative_damage_pct, season_loss_musd, exposure, prelim, prelim_loss, prelim_is_estimate)
    r = int(max(0, min(3, storms_remaining)))
    if r == 0:
        arr = _price_arrays({k: np.array([c[k]]) for k in ISLAND}, np.array([L]), {k: np.array([e[k]]) for k in ISLAND})
        return {k: float(v[0]) for k, v in arr.items()}
    cum, loss, expo = _future(c, L, e, r)
    arr = _price_arrays(cum, loss, expo)
    return {k: float(v.mean()) for k, v in arr.items()}


def fair_and_sigma(cumulative_damage_pct: dict, season_loss_musd: float, storms_remaining: int = 0,
                   exposure: dict | None = None, prelim: dict | None = None, prelim_loss: float | None = None,
                   prelim_is_estimate: bool = False, prelim_sigma: dict | None = None):
    """(fair, sigma, sigma_future) per symbol.
    sigma = uncertainty of our ESTIMATE: prelim-vs-final spread (if a prelim is pending) + model residual.
    sigma_future = spread of the settlement from storms that have not happened yet (0 when all finals are in)."""
    c, L, e = _known_state(cumulative_damage_pct, season_loss_musd, exposure, prelim, prelim_loss, prelim_is_estimate)
    r = int(max(0, min(3, storms_remaining)))
    if r == 0:
        cum = {k: np.array([c[k]]) for k in ISLAND}; expo = {k: np.array([e[k]]) for k in ISLAND}; loss = np.array([L])
    else:
        cum, loss, expo = _future(c, L, e, r)
    arr = _price_arrays(cum, loss, expo)
    fair = {k: float(v.mean()) for k, v in arr.items()}
    future_var = {k: float(v.var()) for k, v in arr.items()}  # settlement spread from storms still to come
    var = {k: 0.0 for k in arr}
    if prelim:  # propagate the prelim-vs-final residual (or the track estimate's spread) by a symmetric perturbation
        pert = {k: (prelim_sigma[k] if prelim_sigma else MODEL["prelim"][k]["diff_std"]) for k in ISLAND}
        lf = MODEL["loss_from_damage"]
        lpert = (sum(lf[k] * pert[k] for k in ISLAND) / 2.0) if prelim_sigma else MODEL["prelim"]["loss"]["diff_std"]
        up = _price_arrays({k: cum[k] + pert[k] for k in ISLAND}, loss + lpert, expo)
        dn = _price_arrays({k: np.maximum(cum[k] - pert[k], 0) for k in ISLAND}, np.maximum(loss - lpert, 0), expo)
        for k in arr:
            var[k] += (0.5 * float((up[k] - dn[k]).mean())) ** 2
    for k in ISLAND:
        var[k] += MODEL["islands"][k]["resid_std_expo"] ** 2
    for k in ("RECON", "MAT"):
        var[k] += MODEL["rebuilders"][k]["resid_std"] ** 2
    var["GEN"] += MODEL["rebuilders"]["GEN"]["resid_std_expo"] ** 2
    for k in ("HAVEN", "REIN"):
        var[k] += MODEL["insurers"][k]["resid_std"] ** 2
    for etf, basket in ETFS.items():
        var[etf] += sum((w ** 2) * var[k] for k, w in basket.items())  # stock model noise, treated as independent
    sigma = {k: float(np.sqrt(max(v, 0.0))) for k, v in var.items()}
    sigma_future = {k: float(np.sqrt(max(v, 0.0))) for k, v in future_var.items()}
    return fair, sigma, sigma_future


def basket_value(etf: str, prices: dict) -> float | None:
    """Value of one ETF unit from a dict of stock prices (None if a leg is missing)."""
    tot = 0.0
    for k, w in ETFS[etf].items():
        if prices.get(k) is None:
            return None
        tot += w * prices[k]
    return tot


if __name__ == "__main__":  # validation against the house prices at the opens
    import sys, time
    sys.path.insert(0, str(HERE / "starter_kit"))
    from history import load
    data = load(HERE / "history")
    t0 = time.perf_counter()
    for _ in range(100):
        fair_values({k: 10.0 for k in ISLAND}, 300.0, storms_remaining=2)
    print(f"fair_values with MC: {(time.perf_counter() - t0) * 10:.2f} ms per call")
    zero = fair_values({}, 0.0, storms_remaining=3)
    print("\nround-1 open: fair (3 storms to come) vs mean house open")
    for k in SYMBOLS:
        print(f"  {k:6} fair {zero[k]:8.2f}  house {MODEL['open_price_mean'][k]:8.2f}  diff {zero[k] - MODEL['open_price_mean'][k]:6.2f}")
    for rnd in (2, 3):
        diffs = {k: [] for k in SYMBOLS}
        for s in data.seasons:
            cum = {k: sum(st["damage_pct"][k] for st in s.storms[:rnd - 1]) for k in ISLAND}
            loss = sum(st["industry_loss_musd"] for st in s.storms[:rnd - 1])
            f = fair_values(cum, loss, storms_remaining=4 - rnd)  # exposure estimated from damage
            j = next(i for i, r in enumerate(s.rounds) if r == rnd)
            for k in SYMBOLS:
                diffs[k].append(f[k] - s.prices[k][j])
        print(f"\nround-{rnd} open: fair - house, mean ± std over 600 seasons")
        for k in SYMBOLS:
            d = np.array(diffs[k]); print(f"  {k:6} {d.mean():6.2f} ± {d.std():5.2f}")
    # settlement check with everything known
    errs = {k: [] for k in SYMBOLS}
    for s in data.seasons:
        cum = {k: sum(st["damage_pct"][k] for st in s.storms) for k in ISLAND}
        f = fair_values(cum, s.season_loss_musd)
        for k in SYMBOLS:
            errs[k].append(f[k] - s.settlement[k])
    print("\nall finals known (exposure estimated from damage): fair - settlement")
    for k in SYMBOLS:
        d = np.array(errs[k]); print(f"  {k:6} {d.mean():6.2f} ± {d.std():5.2f}")
