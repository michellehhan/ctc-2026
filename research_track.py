#!/usr/bin/env python3
"""Live damage estimate from the storm track: a k-nearest-neighbour model over the 1,800 historical storms.

    python3 research_track.py          # builds track_model.npz and prints the accuracy tables

At storm age `a` (real seconds since it formed) the state of a storm is: position, velocity, radius, intensity
(from the latest `storm` observation) and the latest official advisory's hurricane-condition probabilities for
the four islands. For every historical storm we store that state at each 20-second age bucket together with
the storm's FINAL damage per island company, its insured loss and its exposure term. At run time the bot finds
the K most similar historical storms at the same age and uses their mean final damage as the estimate for the
storm in progress, and their spread as the uncertainty.

The tables printed at the end answer the question that matters: at each age, is our settlement estimate
closer to the truth than the house's price at that moment? (Holdout: seasons 481-600.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "starter_kit"))
sys.path.insert(0, str(HERE))
from history import load  # noqa: E402

ISLAND = ["PORT", "ISLE", "AQUA", "GRID", "FERRY"]
ISLES = ["cascadia_prime", "coral_key", "azure_isle", "palm_atoll"]
AGES = np.arange(0, 481, 20)          # 25 buckets over the 480 s storm phase
K = 25
FEATURES = ["x", "y", "vx", "vy", "radius", "intensity"] + [f"p_{i}" for i in ISLES]
NF = len(FEATURES)


def storm_states(feed_round: list[dict], ages=AGES) -> np.ndarray:
    """(len(ages), NF) feature matrix for one storm; NaN where the information does not exist yet."""
    obs = [m for m in feed_round if m["type"] == "storm"]
    adv = [m for m in feed_round if m["type"] == "advisory" and m.get("source") == "official"]
    out = np.full((len(ages), NF), np.nan)
    oi = ai = 0
    last_obs = last_adv = None
    for r, a in enumerate(ages):
        while oi < len(obs) and obs[oi]["age"] <= a:
            last_obs = obs[oi]; oi += 1
        while ai < len(adv) and adv[ai]["age"] <= a:
            last_adv = adv[ai]; ai += 1
        if last_obs is not None:
            out[r, :6] = [last_obs["x"], last_obs["y"], last_obs["vx"], last_obs["vy"], last_obs["radius"], last_obs.get("intensity", 1.0)]
        if last_adv is not None:
            hc = last_adv.get("hurricane_conditions", {})
            out[r, 6:] = [hc.get(i, 0.0) for i in ISLES]
    return out


def exposure_of_round(feed_round, case, kinds):
    obs = [m for m in feed_round if m["type"] == "storm"]
    expo = {}
    for k in ISLAND:
        t = 0.0
        for site in case["companies"][k]["sites"]:
            for m in obs:
                if np.hypot(m["x"] - site["x"], m["y"] - site["y"]) < m["radius"]:
                    t += 2.0 * site["weight"] * (m.get("intensity", 1.0) if kinds[k] == "expo_i" else 1.0)
        expo[k] = t
    return expo


class TrackModel:
    """k-NN estimate of a storm's final damage from its state at a given age. Loads track_model.npz."""

    def __init__(self, path=HERE / "track_model.npz"):
        d = np.load(path)
        self.X = d["X"]            # (n_storms, n_ages, NF)
        self.Y = d["Y"]            # (n_storms, 11): 5 damage, 5 expo, loss
        self.mu = d["mu"]; self.sd = d["sd"]   # per age, per feature
        self.ages = d["ages"]
        self.prior_dmg = self.Y[:, :5].mean(axis=0)

    def estimate(self, age: float, state: np.ndarray, k: int = K):
        """-> (damage dict, damage sigma dict, expo dict, loss) for a storm at `age` with feature vector `state`."""
        r = int(np.clip(np.searchsorted(self.ages, age, side="right") - 1, 0, len(self.ages) - 1))
        s = state.copy()
        have = ~np.isnan(s)
        Xr = self.X[:, r, :]
        ok = ~np.isnan(Xr).any(axis=1) if have.all() else ~np.isnan(Xr[:, have]).any(axis=1)
        if ok.sum() < k or not have[:6].all():
            d = {kk: float(self.prior_dmg[i]) for i, kk in enumerate(ISLAND)}
            sg = {kk: float(self.Y[:, i].std()) for i, kk in enumerate(ISLAND)}
            ex = {kk: float(self.Y[:, 5 + i].mean()) for i, kk in enumerate(ISLAND)}
            return d, sg, ex, float(self.Y[:, 10].mean())
        z = (Xr[ok][:, have] - self.mu[r, have]) / self.sd[r, have]
        zq = (s[have] - self.mu[r, have]) / self.sd[r, have]
        dist = np.sqrt(((z - zq) ** 2).sum(axis=1))
        idx = np.argpartition(dist, k)[:k]
        Yn = self.Y[ok][idx]
        d = {kk: float(Yn[:, i].mean()) for i, kk in enumerate(ISLAND)}
        sg = {kk: float(Yn[:, i].std()) for i, kk in enumerate(ISLAND)}
        ex = {kk: float(Yn[:, 5 + i].mean()) for i, kk in enumerate(ISLAND)}
        return d, sg, ex, float(Yn[:, 10].mean())


def build():
    data = load(HERE / "history")
    case = json.load(open(HERE / "case.json"))
    model = json.load(open(HERE / "model.json"))
    kinds = {k: model["islands"][k]["expo_kind"] for k in ISLAND}
    S = data.seasons
    X, Y, meta = [], [], []
    print("building per-storm states...", flush=True)
    for s in S:
        for st in s.storms:
            fr = [m for m in s.feed if m["round"] == st["round"]]
            X.append(storm_states(fr))
            ex = exposure_of_round(fr, case, kinds)
            Y.append([st["damage_pct"][k] for k in ISLAND] + [ex[k] for k in ISLAND] + [st["industry_loss_musd"]])
            meta.append((s.number, st["round"]))
    X = np.array(X); Y = np.array(Y)
    train = np.array([m[0] <= 480 for m in meta])
    mu = np.nanmean(X[train], axis=0); sd = np.nanstd(X[train], axis=0) + 1e-9
    np.savez(HERE / "track_model.npz", X=X[train], Y=Y[train], mu=mu, sd=sd, ages=AGES)
    print(f"saved track_model.npz: {train.sum()} training storms, {(~train).sum()} holdout storms")

    # ------------------------------------------------------------------ evaluation on holdout seasons
    from valuation import fair_values
    tm = TrackModel()
    hold = [s for s in S if s.number > 480]
    print("\nHoldout accuracy of the damage estimate vs the unconditional mean, by storm age (std of error in damage points):")
    print(f"{'age':>4} " + " ".join(f"{k:>12}" for k in ISLAND))
    for r, a in enumerate(AGES[::3]):
        err = {k: [] for k in ISLAND}; base = {k: [] for k in ISLAND}
        for s in hold:
            for st in s.storms:
                i = [j for j, m in enumerate(meta) if m == (s.number, st["round"])][0]
                d, sg, ex, L = tm.estimate(a, X[i, 3 * r])
                for k in ISLAND:
                    err[k].append(d[k] - st["damage_pct"][k]); base[k].append(tm.prior_dmg[ISLAND.index(k)] - st["damage_pct"][k])
        print(f"{a:4d} " + " ".join(f"{np.std(err[k]):5.1f}/{np.std(base[k]):5.1f}" for k in ISLAND))

    print("\nSettlement prediction during the storm: std(settlement - OUR fair) vs std(settlement - HOUSE price), holdout seasons.")
    print("Our fair uses the k-NN estimate for the storm in progress. Smaller is better; 'ours < house' is the edge.")
    syms = ["PORT", "ISLE", "AQUA", "GRID", "FERRY", "RECON", "MAT", "GEN", "HAVEN", "REIN", "ARCH", "REBLD"]
    for rnd in (1, 2, 3):
        print(f"\nround {rnd}:  age  " + " ".join(f"{k:>11}" for k in syms))
        for a in (60, 120, 180, 240, 360, 480):
            ours = {k: [] for k in syms}; house = {k: [] for k in syms}
            for s in hold:
                st = s.storms[rnd - 1]
                i = [j for j, m in enumerate(meta) if m == (s.number, rnd)][0]
                r = int(np.searchsorted(AGES, a, side="right") - 1)
                d, sg, ex, L = tm.estimate(a, X[i, r])
                cum = {k: sum(x["damage_pct"][k] for x in s.storms[:rnd - 1]) for k in ISLAND}
                loss = sum(x["industry_loss_musd"] for x in s.storms[:rnd - 1])
                f = fair_values(cum, loss, storms_remaining=3 - rnd, prelim=d, prelim_loss=L, prelim_is_estimate=True)
                # house price at that age: storm forms 60 s after the open -> sample index (60 + a) / 2
                j0 = next(i2 for i2, rr in enumerate(s.rounds) if rr == rnd)
                j = j0 + (60 + a) // 2
                for k in syms:
                    ours[k].append(s.settlement[k] - f[k]); house[k].append(s.settlement[k] - s.prices[k][j])
            print(f"          {a:4d} " + " ".join(f"{np.std(ours[k]):5.2f}/{np.std(house[k]):5.2f}" for k in syms))


if __name__ == "__main__":
    build()
