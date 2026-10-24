#!/usr/bin/env python3
"""Hurricane Season trading bot (Systematic Equities case).

    export CTC_API_KEY=...            # bot key from the Team page
    python3 bot.py                    # live: connects to the exchange (testnet or competition, see BASE_URL)
    python3 bot.py --dry-run 7        # replay history/season-007 through the full strategy; no network
    python3 bot.py --dry-run 1-40     # replay a range of seasons, print the PnL distribution
    python3 bot.py --size 0.2         # scale every position size down (testnet runs with small sizes)

Strategies (flags in CONFIG):
  A  on every damage report (prelim/final) and every RECHECK_SECS after it: fair values from valuation.py,
     buy when the ask is below fair - edge, sell when the bid is above fair + edge, sized by confidence.
  B  ETF arbitrage every 2 s: ETF quote vs basket of stock quotes, then create/redeem to flatten.
  C  pre-report lean after 2+ consecutive official "direct" calls (off by default; see RUNBOOK.md).
Announcements are printed loudly and never traded automatically: use manual.py.

All orders are limit orders that cross the spread by CROSS_FRAC. Position limits include resting orders.
The bot stays under RATE_BUDGET_PER_MIN order actions (team limit is 180/min) to leave room for manual.py.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import math
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exchange import Exchange, ExchangeError, load_key  # noqa: E402
from valuation import ETFS, ISLAND, LIMITS, MODEL, STOCKS, SYMBOLS, fair_and_sigma  # noqa: E402

# =============================================================== CONFIG (edit here) ===================
CONFIG = dict(
    BASE_URL="https://cornelltradingcompetition.org",
    WS_URL=None,                # None -> derived from BASE_URL (wss://.../ws/v1/market-data)
    STRAT_A=True,
    STRAT_B=True,
    STRAT_C=False,              # pre-report lean on consecutive "direct" advisories
    # --- Strategy A: edge and sizing
    EDGE_K=1.5,                 # required edge = EDGE_K * sigma_model + FUTURE_K * sigma_future + MIN_EDGE
    FUTURE_K=0.05,              # extra edge per $ of settlement spread from storms still to come (rounds 1-2)
    ANCHOR_WINDOW_SECS=60.0,    # rounds 1-2 (if sizes above are > 0): trade a report only this long after it, against the
                                # house's PRE-report mid plus R12_DELTA_SLOPE x our change in fair value ("delta" mode).
    R12_DELTA_SLOPE=0.25,       # share of our fair-value change the house moves on a round-1/2 final report (research.py)
    MIN_EDGE_FRAC=0.002,        # ... plus 0.2% of the price
    MIN_EDGE_ABS=0.05,          # ... and at least 5 cents
    SIZE_FINAL_R3=1.0,          # fraction of the position limit after the FINAL report in round 3 (settlement known)
    SIZE_FINAL_R12=0.0,         # after the final report in rounds 1-2 (0 = off: research shows the house already prices
                                #   the storm track, and its reaction to a report is only ~25% of ours -> no reliable edge)
    SIZE_PRELIM_R12=0.0,        # after a preliminary report in rounds 1-2 (0 = off, same reason: house moves ~5% of ours)
    SIZE_PRELIM=0.3,            # after the preliminary report in round 3 (settlement nearly known)
    SIZE_PRE_REPORT=0.0,        # while the storm is running and no report is out (0 = no Strategy A trading then)
    FULL_SIZE_AT_EDGE_MULT=2.0,  # full size when |edge| >= this multiple of the required edge; linear below
    RECHECK_SECS=2.0,           # re-run Strategy A against the live quotes this often after a report
    CROSS_FRAC=0.001,           # limit price crosses the best quote by this fraction
    ORDER_TTL_SECS=8.0,         # cancel our resting orders older than this
    QUOTE_STALE_SECS=15.0,      # ignore quotes older than this (live mode)
    # --- Strategy B: ETF arbitrage
    ETF_ARB_THRESHOLD=0.60,     # $ per ETF unit after crossing every spread
    ETF_ARB_UNITS=10,           # ETF units per arb
    ETF_ARB_CROSS_SHARE=0.4,    # the ETF leg's limit crosses the quote by this share of the apparent profit
    ETF_ARB_MAX_PER_MIN=6,
    ETF_ARB_EVERY_SECS=2.0,
    # --- Strategy C
    STRAT_C_SIZE=0.3,
    STRAT_C_MIN_CONSECUTIVE=2,
    STRAT_C_MAX_ADVISORY=4,     # only act on advisory numbers <= this (later calls add little)
    STRAT_C_MIN_PRECISION=0.75,  # from model.json advisory table
    # --- plumbing
    RATE_BUDGET_PER_MIN=150,    # order actions per minute we allow ourselves (team limit 180)
    POLL_SECS=3.0,              # GET /positions and /orders/open
    PNL_EVERY_SECS=10.0,
    TICK=0.01,
    LOG_FILE="bot.log",
    STATE_FILE="bot_state.json",
    # --- dry run (replay of history/season-NNN)
    DRY_RUN_SPREAD_FRAC=0.003,  # full bid-ask spread of the synthetic house quotes as a fraction of price (unknown on testnet)
    DRY_RUN_LATENCY_SAMPLES=1,  # our order reaches the book this many 2-second samples after we send it
)
# ======================================================================================================

ROUND_SECS = 780.0
CASE = json.load(open(HERE / "case.json"))
SITES = {k: [(s["x"], s["y"], s["weight"]) for s in CASE["companies"][k]["sites"]] for k in ISLAND}
EXPO_KIND = {k: MODEL["islands"][k]["expo_kind"] for k in ISLAND}
HAVEN_TOP = MODEL["insurers"]["HAVEN"]["attach"] + MODEL["insurers"]["HAVEN"]["layer"]
OBS_DT = 2.0

log = logging.getLogger("bot")


def setup_logging(path: Path, console_level=logging.INFO):
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path, mode="a"); fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.setLevel(console_level); root.addHandler(ch)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def round_tick(price: float, side: str, tick: float) -> float:
    q = price / tick
    return round((math.ceil(q) if side == "buy" else math.floor(q)) * tick, 2)


class RateLimiter:
    """Sliding one-minute window over order actions (place, cancel, cancel-all, create, redeem)."""

    def __init__(self, budget: int, clock):
        self.budget, self.clock, self.stamps = budget, clock, collections.deque()
        self.lock = threading.Lock()

    def used(self) -> int:
        now = self.clock()
        with self.lock:
            while self.stamps and now - self.stamps[0] > 60.0:
                self.stamps.popleft()
            return len(self.stamps)

    def take(self, n: int = 1, reserve: int = 0) -> bool:
        if self.used() + n > self.budget - reserve:
            return False
        with self.lock:
            now = self.clock()
            for _ in range(n):
                self.stamps.append(now)
        return True


# ======================================================================================================
class ReplayExchange:
    """Dry-run stand-in for Exchange: fills our crossing limit orders against synthetic house quotes built
    from market.csv (mid +- half spread), with a small latency. Depth is unlimited (optimistic)."""

    def __init__(self, spread_frac: float, latency: int, tick: float):
        self.spread_frac, self.latency, self.tick = spread_frac, latency, tick
        self.quotes: dict[str, tuple[float, float, float]] = {}  # sym -> (bid, ask, mid)
        self.pos: dict[str, int] = {s: 0 for s in SYMBOLS}
        self.cash = 0.0
        self.orders: dict[str, dict] = {}
        self.next_id = 1
        self.idx = -1
        self.fills: list[dict] = []
        self.tag_cash: dict[str, float] = collections.defaultdict(float)
        self.tag_pos: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        self.actions = 0

    # -- market
    def on_sample(self, idx: int, row: dict):
        self.idx = idx
        for s in SYMBOLS:
            mid = float(row[s]); half = max(self.tick, 0.5 * self.spread_frac * mid)
            self.quotes[s] = (round(mid - half, 2), round(mid + half, 2), mid)
        for oid in list(self.orders):
            o = self.orders[oid]
            if o["active_from"] > idx:
                continue
            bid, ask, _ = self.quotes[o["symbol"]]
            if o["side"] == "buy" and o["price"] >= ask:
                self._fill(o, ask)
            elif o["side"] == "sell" and o["price"] <= bid:
                self._fill(o, bid)

    def _fill(self, o: dict, px: float):
        q = o["quantity"]; sign = 1 if o["side"] == "buy" else -1
        self.pos[o["symbol"]] += sign * q
        self.cash -= sign * q * px
        self.tag_cash[o["tag"]] -= sign * q * px
        self.tag_pos[o["tag"]][o["symbol"]] += sign * q
        self.fills.append({"idx": self.idx, "symbol": o["symbol"], "side": o["side"], "quantity": q, "price": px, "tag": o["tag"], "limit": o["price"]})
        log.info("FILL  %-5s %-4s %4d %-6s @ %.2f (limit %.2f) [%s]", o["symbol"], o["side"], q, "", px, o["price"], o["tag"])
        del self.orders[o["id"]]

    # -- Exchange API
    def place_order(self, symbol, side, quantity, price=None, tag="?"):
        self.actions += 1
        if price is None:
            bid, ask, _ = self.quotes[symbol]
            price = ask * 1.01 if side == "buy" else bid * 0.99
        oid = str(self.next_id); self.next_id += 1
        o = {"id": oid, "symbol": symbol, "side": side, "quantity": int(quantity), "price": float(price), "tag": tag,
             "active_from": self.idx + self.latency}
        self.orders[oid] = o
        if self.latency == 0:
            bid, ask, _ = self.quotes[symbol]
            if (side == "buy" and price >= ask) or (side == "sell" and price <= bid):
                self._fill(o, ask if side == "buy" else bid)
                return {"order_id": oid, "status": "filled", "filled_quantity": int(quantity)}
        return {"order_id": oid, "status": "accepted", "filled_quantity": 0}

    def cancel(self, order_id):
        self.actions += 1
        self.orders.pop(str(order_id), None)
        return {"status": "cancelled"}

    def cancel_all(self, symbol=None):
        self.actions += 1
        for oid in list(self.orders):
            if symbol is None or self.orders[oid]["symbol"] == symbol:
                del self.orders[oid]
        return {"status": "cancelled"}

    def open_orders(self):
        return [dict(o) for o in self.orders.values()]

    def positions(self):
        return dict(self.pos)

    def time(self):
        return {"state": "running"}

    def etf_create(self, symbol, quantity, tag="B"):
        self.actions += 1
        q = int(quantity)
        self.pos[symbol] += q; self.tag_pos[tag][symbol] += q
        for k, w in ETFS[symbol].items():
            self.pos[k] -= w * q; self.tag_pos[tag][k] -= w * q
        return {"status": "ok"}

    def etf_redeem(self, symbol, quantity, tag="B"):
        self.actions += 1
        q = int(quantity)
        self.pos[symbol] -= q; self.tag_pos[tag][symbol] -= q
        for k, w in ETFS[symbol].items():
            self.pos[k] += w * q; self.tag_pos[tag][k] += w * q
        return {"status": "ok"}

    # -- parsers (same interface as Exchange)
    @staticmethod
    def parse_positions(payload):
        return dict(payload)

    @staticmethod
    def parse_open_orders(payload):
        return [{"id": o["id"], "symbol": o["symbol"], "side": o["side"], "quantity": o["quantity"], "price": o["price"]} for o in payload]

    @staticmethod
    def parse_order_response(resp):
        return resp.get("order_id"), resp.get("status", "accepted"), resp.get("filled_quantity", 0)


# ======================================================================================================
class Bot:
    def __init__(self, ex, clock, cfg: dict, dry_run: bool, size_scale: float = 1.0):
        self.ex, self.clock, self.cfg, self.dry_run, self.size_scale = ex, clock, cfg, dry_run, size_scale
        self.rate = RateLimiter(cfg["RATE_BUDGET_PER_MIN"], clock)
        self.pool = None if dry_run else ThreadPoolExecutor(max_workers=8)
        self.lock = threading.RLock()
        self.counts = collections.Counter()
        self.reset_season()
        self.quotes: dict[str, dict] = {}
        self.pos: dict[str, int] = {s: 0 for s in SYMBOLS}
        self.orders: dict[str, dict] = {}      # our open orders: id -> {symbol, side, quantity, price, ts, tag}
        self.cash_est = 0.0                    # approximate cash from fills we know about
        self.running = True                    # competition state (live)
        self.last = collections.defaultdict(lambda: -1e9)
        self.pending_conv: list[dict] = []     # ETF conversions to do after the legs fill
        self.mid_hist: dict[str, collections.deque] = {s: collections.deque() for s in SYMBOLS}
        self.arb_stamps = collections.deque()
        self.connected = False

    # ------------------------------------------------------------------ season state
    def reset_season(self):
        self.round = 0
        self.phase = "pre_storm"
        self.storm = ""
        self.cum = {k: 0.0 for k in ISLAND}        # final damage so far
        self.loss = 0.0                            # final insured loss so far
        self.expo = {k: 0.0 for k in ISLAND}       # outage exposure so far (all storms incl. current)
        self.prelim: dict | None = None
        self.prelim_loss: float | None = None
        self.finals_done: set[int] = set()
        self.reports_seen: set[tuple] = set()
        self.seen_seq: set[int] = set()
        self.last_seq = -1
        self.advisory: dict | None = None
        self.direct_streak = 0
        self.c_done = False
        self.last_report_round = 0
        self.last_report_stage = None
        self.fair: dict = {}
        self.sigma: dict = {}
        self.sigma_future: dict = {}
        self.fair_prev: dict = {}
        self.anchor_mid: dict = {}
        self.anchor_ts = -1e9
        self.compute_fairs()

    @property
    def storms_remaining(self) -> int:
        return max(0, 3 - len(self.finals_done) - (1 if self.prelim is not None else 0))

    # ------------------------------------------------------------------ feed
    def handle_frame(self, frame: dict):
        t = frame.get("type")
        msg = frame.get("message")
        if t == "news" and isinstance(msg, dict):
            self.on_news(msg)
        elif t in ("quote", "quotes", "bbo") or (isinstance(msg, dict) and "bid" in msg and "symbol" in msg):
            self.on_quote(msg if isinstance(msg, dict) and "symbol" in msg else frame)
        elif t == "_connected":
            self.connected = True
        elif t == "_disconnected":
            self.connected = False
        else:
            self.counts[f"frame:{t}"] += 1

    def on_quote(self, q: dict):
        sym = q.get("symbol")
        if sym not in LIMITS:
            return
        bid = q.get("bid", q.get("best_bid", q.get("bid_price")))
        ask = q.get("ask", q.get("best_ask", q.get("ask_price")))
        if isinstance(bid, dict):
            bid = bid.get("price")
        if isinstance(ask, dict):
            ask = ask.get("price")
        now = self.clock()
        self.quotes[sym] = {"bid": float(bid) if bid is not None else None, "ask": float(ask) if ask is not None else None, "ts": now}
        if bid is not None and ask is not None:
            h = self.mid_hist[sym]
            h.append((now, 0.5 * (float(bid) + float(ask))))
            while h and now - h[0][0] > 60.0:
                h.popleft()
        self.counts["quote"] += 1

    def on_news(self, m: dict):
        t = m.get("type")
        self.counts[t] += 1
        seq = m.get("seq")
        rnd = m.get("round")
        # --- new season? (testnet rolls to the next season: round goes back to 1, seq restarts)
        if rnd is not None and self.round and (rnd < self.round or (seq is not None and seq < self.last_seq - 500)):
            log.warning("NEW SEASON detected (round %s after %s, seq %s): resetting state", rnd, self.round, seq)
            self.reset_season()
            self.pos = {s: 0 for s in SYMBOLS}; self.orders = {}; self.cash_est = 0.0; self.pending_conv = []
        if seq is not None:
            if seq in self.seen_seq:
                self.counts["dup"] += 1
                return
            self.seen_seq.add(seq)
            if len(self.seen_seq) > 20000:
                self.seen_seq = set(sorted(self.seen_seq)[-5000:])
            self.last_seq = max(self.last_seq, seq)
        if rnd is not None:
            self.round = int(rnd)
        if m.get("storm"):
            self.storm = m["storm"]
        if t == "storm_formed":
            self.phase = "storm"; self.direct_streak = 0; self.c_done = False
            self.compute_fairs()
            log.info("=== round %s: %s formed === fair %s", self.round, self.storm, {k: round(v, 2) for k, v in self.fair.items()})
        elif t == "storm":
            self._on_obs(m)
        elif t == "storm_dissipated":
            self.phase = "aftermath"
        elif t == "advisory":
            if m.get("source") == "official":
                self.advisory = m
                self.direct_streak = self.direct_streak + 1 if m.get("outlook") == "direct" else 0
                log.info("ADVISORY %s: %s | %s", m.get("number"), m.get("outlook"), m.get("headline", ""))
                self.strategy_c(m)
            else:
                log.debug("wire: %s", m.get("headline", ""))
        elif t == "impact":
            log.info("IMPACT: %s", m.get("headline", ""))
        elif t == "damage_report":
            self._on_report(m, from_snapshot=False)
        elif t in ("announcement", "news"):
            self._announce(m)
        elif t == "snapshot":
            self._on_snapshot(m)
        elif t == "track":
            pass
        else:
            log.debug("message %s: %s", t, m.get("headline", ""))

    def _on_obs(self, m: dict):
        x, y, r, inten = m.get("x"), m.get("y"), m.get("radius"), m.get("intensity", 1.0)
        if x is None or y is None or r is None:
            return
        for k, sites in SITES.items():
            for sx, sy, w in sites:
                if math.hypot(x - sx, y - sy) < r:
                    self.expo[k] += OBS_DT * w * (inten if EXPO_KIND[k] == "expo_i" else 1.0)

    def _on_report(self, m: dict, from_snapshot: bool):
        rnd = int(m.get("round", self.round) or self.round)
        stage = m.get("stage")
        key = (rnd, stage)
        if key in self.reports_seen:
            return
        self.reports_seen.add(key)
        dmg = m.get("damage_pct", {}) or {}
        loss = float(m.get("industry_loss_musd", 0.0) or 0.0)
        # anchor: our fair and the house mid from just before this report (for "delta" mode in rounds 1-2)
        self.compute_fairs()
        self.fair_prev = dict(self.fair)
        now = self.clock()
        self.anchor_mid = {}
        for s in SYMBOLS:
            older = [mid for ts, mid in self.mid_hist[s] if ts <= now - 1.0]
            cur = self.mid(s)
            self.anchor_mid[s] = older[-1] if older else cur
        self.anchor_ts = now
        if stage == "preliminary":
            if rnd in self.finals_done:
                return
            self.prelim = {k: float(dmg.get(k, 0.0)) for k in ISLAND}
            self.prelim_loss = loss
        elif stage == "final":
            for k in ISLAND:
                self.cum[k] += float(dmg.get(k, 0.0))
            self.loss += loss
            self.finals_done.add(rnd)
            self.prelim = None; self.prelim_loss = None
        else:
            return
        self.last_report_round, self.last_report_stage = rnd, stage
        log.info("REPORT %s round %d%s: loss %.0f (season %.0f) dmg %s | cum %s", stage.upper(), rnd,
                 " (from snapshot)" if from_snapshot else "", loss, self.loss, dmg, {k: round(v, 1) for k, v in self.cum.items()})
        self.compute_fairs()
        self.strategy_a(stage)

    def _on_snapshot(self, m: dict):
        if m.get("phase"):
            self.phase = m["phase"]
        if m.get("round"):
            self.round = int(m["round"])
        for rep in m.get("damage", []) or []:
            if isinstance(rep, dict) and rep.get("stage"):
                self._on_report(rep, from_snapshot=True)
        if isinstance(m.get("advisory"), dict):
            self.advisory = m["advisory"]
        if m.get("last_seq") is not None:
            self.last_seq = max(self.last_seq, int(m["last_seq"]))
        log.debug("snapshot: round %s phase %s", self.round, self.phase)

    def _announce(self, m: dict):
        banner = "#" * 100
        text = f"\a\n{banner}\n!!! {m.get('type', '').upper()} round {m.get('round')} symbols {m.get('symbols')}\n!!! {m.get('headline', '')}\n!!! {json.dumps(m)[:800]}\n{banner}\n\a"
        log.warning(text)
        print(text, flush=True)

    # ------------------------------------------------------------------ valuation
    def compute_fairs(self):
        try:
            self.fair, self.sigma, self.sigma_future = fair_and_sigma(
                self.cum, self.loss, storms_remaining=self.storms_remaining, exposure=self.expo,
                prelim=self.prelim, prelim_loss=self.prelim_loss)
        except Exception:
            log.exception("fair value computation failed")

    def mid(self, sym: str):
        q = self.quotes.get(sym)
        if not q or q["bid"] is None or q["ask"] is None:
            return None
        return 0.5 * (q["bid"] + q["ask"])

    def live_quote(self, sym: str):
        q = self.quotes.get(sym)
        if not q or q["bid"] is None or q["ask"] is None:
            return None
        if not self.dry_run and self.clock() - q["ts"] > self.cfg["QUOTE_STALE_SECS"]:
            return None
        return q

    # ------------------------------------------------------------------ orders
    def resting(self, sym: str, side: str) -> int:
        return sum(o["quantity"] for o in self.orders.values() if o["symbol"] == sym and o["side"] == side)

    def room(self, sym: str, side: str) -> int:
        """Shares we may still send on this side without breaching the limit (resting orders count)."""
        lim = LIMITS[sym]; p = self.pos.get(sym, 0)
        return max(0, lim - p - self.resting(sym, "buy")) if side == "buy" else max(0, lim + p - self.resting(sym, "sell"))

    def send(self, sym: str, side: str, qty: int, price: float, tag: str, reserve: int = 0) -> bool:
        qty = int(min(qty, self.room(sym, side)))
        if qty <= 0:
            return False
        if not self.running:
            log.info("skip %s %s %d: competition not running", side, sym, qty)
            return False
        if not self.rate.take(1, reserve):
            log.warning("RATE LIMIT: skip %s %s %d (%d actions in last minute)", side, sym, qty, self.rate.used())
            return False
        price = round_tick(price, side, self.cfg["TICK"])
        oid = f"tmp{time.time_ns()}"
        if self.dry_run:
            stage = self.last_report_stage if self.last_report_round == self.round else "pre"
            tag = f"{tag}|r{self.round}/{stage}|{sym}"
        with self.lock:  # reserve the room immediately so a burst does not overshoot the limit
            self.orders[oid] = {"symbol": sym, "side": side, "quantity": qty, "price": price, "ts": self.clock(), "tag": tag}
        log.info("ORDER %-5s %-4s %4d @ %.2f [%s] fair %.2f pos %+d", sym, side, qty, price, tag, self.fair.get(sym, float("nan")), self.pos.get(sym, 0))
        if self.pool:
            self.pool.submit(self._send_now, oid, sym, side, qty, price, tag)
        else:
            self._send_now(oid, sym, side, qty, price, tag)
        return True

    def _send_now(self, tmp_id, sym, side, qty, price, tag):
        try:
            resp = self.ex.place_order(sym, side, qty, price, tag=tag) if self.dry_run else self.ex.place_order(sym, side, qty, price)
            real_id, status, filled = self.ex.parse_order_response(resp)
            with self.lock:
                o = self.orders.pop(tmp_id, None)
                if o is None:
                    return
                if filled:
                    self._apply_fill(sym, side, filled, price, tag)
                    o["quantity"] -= filled
                if o["quantity"] > 0 and status not in ("filled", "rejected", "cancelled", "canceled") and real_id:
                    self.orders[real_id] = o
            log.debug("order resp %s %s %s", sym, status, json.dumps(resp)[:300])
        except ExchangeError as exc:
            with self.lock:
                self.orders.pop(tmp_id, None)
            log.error("ORDER REJECTED %s %s %d @ %.2f: %s", sym, side, qty, price, exc)
        except Exception:
            with self.lock:
                self.orders.pop(tmp_id, None)
            log.exception("order send failed")

    def _apply_fill(self, sym, side, qty, price, tag):
        sign = 1 if side == "buy" else -1
        self.pos[sym] = self.pos.get(sym, 0) + sign * qty
        self.cash_est -= sign * qty * price
        log.info("fill  %-5s %-4s %4d @ %.2f [%s] -> pos %+d", sym, side, qty, price, tag, self.pos[sym])

    def cancel_order(self, oid: str):
        if oid.startswith("tmp"):
            return
        if not self.rate.take(1):
            return
        o = self.orders.pop(oid, None)
        try:
            self.ex.cancel(oid)
            log.info("CANCEL %s %s", oid, o)
        except ExchangeError as exc:
            log.warning("cancel %s failed: %s", oid, exc)

    def action(self, fn, *args, **kw):
        """Run an order action (create/redeem/cancel-all) under the rate limit; never raises."""
        if not self.rate.take(1):
            log.warning("RATE LIMIT: skip %s", getattr(fn, "__name__", fn))
            return None
        try:
            return fn(*args, **kw)
        except ExchangeError as exc:
            log.error("%s failed: %s", getattr(fn, "__name__", fn), exc)
        except Exception:
            log.exception("%s failed", getattr(fn, "__name__", fn))
        return None

    # ------------------------------------------------------------------ polling / reconciliation
    def poll(self):
        try:
            pos = self.ex.parse_positions(self.ex.positions())
            with self.lock:
                for sym in SYMBOLS:
                    new = int(pos.get(sym, 0)); old = self.pos.get(sym, 0)
                    if new != old:  # a fill we did not see: assume our order price on that side, else mid
                        side = "buy" if new > old else "sell"
                        ours = [o for o in self.orders.values() if o["symbol"] == sym and o["side"] == side]
                        px = ours[0]["price"] if ours else (self.mid(sym) or 0.0)
                        self.cash_est -= (new - old) * px
                        log.info("poll: %s position %+d -> %+d (fill assumed @ %.2f)", sym, old, new, px)
                        self.pos[sym] = new
            oo = self.ex.parse_open_orders(self.ex.open_orders())
            with self.lock:
                known = {oid: o for oid, o in self.orders.items()}
                self.orders = {oid: o for oid, o in known.items() if oid.startswith("tmp")}
                for o in oo:
                    k = known.get(o["id"], {})
                    self.orders[o["id"]] = {"symbol": o["symbol"], "side": o["side"], "quantity": o["quantity"], "price": o["price"],
                                            "ts": k.get("ts", self.clock()), "tag": k.get("tag", "ext")}
        except ExchangeError as exc:
            log.warning("poll failed: %s", exc)
        except Exception:
            log.exception("poll failed")
        if not self.dry_run:
            try:
                t = self.ex.time()
                state = str(t.get("state", "running")).lower()
                was = self.running
                self.running = state in ("running", "open", "active", "live")
                if was != self.running:
                    log.warning("competition state: %s -> trading %s", state, "ON" if self.running else "OFF")
                self.counts["clock:" + state] += 1
            except Exception as exc:
                log.debug("time poll failed: %s", exc)

    def expire_orders(self):
        now = self.clock()
        for oid, o in list(self.orders.items()):
            if now - o["ts"] > self.cfg["ORDER_TTL_SECS"]:
                self.cancel_order(oid)

    # ------------------------------------------------------------------ strategies
    def size_context(self, trigger: str):
        """Fraction of the limit to use for Strategy A in the current situation."""
        c = self.cfg
        stage = self.last_report_stage if self.last_report_round == self.round else None
        rnd = self.round
        if stage == "final":
            return c["SIZE_FINAL_R3"] if rnd >= 3 else c["SIZE_FINAL_R12"]
        if stage == "preliminary":
            return c["SIZE_PRELIM"] if rnd >= 3 else c["SIZE_PRELIM_R12"]
        return c["SIZE_PRE_REPORT"]

    def strategy_a(self, trigger: str):
        if not self.cfg["STRAT_A"] or not self.fair:
            return
        c = self.cfg
        frac = self.size_context(trigger) * self.size_scale
        if frac <= 0:
            return
        level_mode = self.round >= 3 and self.last_report_round >= 3  # settlement (nearly) known: trade the level
        if not level_mode and (self.clock() - self.anchor_ts > c["ANCHOR_WINDOW_SECS"] or not self.fair_prev):
            return
        for sym in SYMBOLS:
            q = self.live_quote(sym)
            if q is None:
                continue
            fair = self.fair[sym]
            if not level_mode:
                a = self.anchor_mid.get(sym)
                if a is None or sym not in self.fair_prev:
                    continue
                fair = a + c["R12_DELTA_SLOPE"] * (self.fair[sym] - self.fair_prev[sym])  # house's pre-report mid + its likely move
            req = c["EDGE_K"] * self.sigma.get(sym, 0.0) + c["FUTURE_K"] * self.sigma_future.get(sym, 0.0) \
                + max(c["MIN_EDGE_ABS"], c["MIN_EDGE_FRAC"] * fair)
            if q["ask"] < fair - req:
                side, edge, px = "buy", fair - q["ask"], q["ask"] * (1 + c["CROSS_FRAC"])
            elif q["bid"] > fair + req:
                side, edge, px = "sell", q["bid"] - fair, q["bid"] * (1 - c["CROSS_FRAC"])
            else:
                continue
            conf = min(1.0, edge / (c["FULL_SIZE_AT_EDGE_MULT"] * req))
            target = int(round(LIMITS[sym] * frac * conf))
            pos = self.pos.get(sym, 0)
            want = target - pos if side == "buy" else target + pos  # shares to add on this side
            want -= self.resting(sym, side)
            if want <= 0:
                continue
            log.info("A/%s %s: %s %.2f sigma %.2f req %.2f quote %.2f/%.2f edge %.2f -> %s %d", trigger, sym,
                     "fair" if level_mode else "anchor+delta", fair, self.sigma.get(sym, 0), req, q["bid"], q["ask"], edge, side, want)
            self.send(sym, side, want, px, "A")

    def strategy_b(self):
        """ETF arbitrage, ETF leg first: buy (sell) the ETF when it is cheap (rich) against the basket of stock
        quotes; once that fills, redeem (create) and sell (buy) the stocks. No stock exposure if the ETF leg misses."""
        if not self.cfg["STRAT_B"]:
            return
        c = self.cfg; now = self.clock()
        while self.arb_stamps and now - self.arb_stamps[0] > 60:
            self.arb_stamps.popleft()
        if len(self.arb_stamps) >= c["ETF_ARB_MAX_PER_MIN"]:
            return
        for etf, basket in ETFS.items():
            if any(pc["etf"] == etf for pc in self.pending_conv):
                continue  # one arb per ETF in flight
            q = self.live_quote(etf)
            legs = {k: self.live_quote(k) for k in basket}
            if q is None or any(v is None for v in legs.values()):
                continue
            basket_bid = sum(w * legs[k]["bid"] for k, w in basket.items())
            basket_ask = sum(w * legs[k]["ask"] for k, w in basket.items())
            p_buy_etf = basket_bid - q["ask"]     # buy ETF, redeem into stocks, sell stocks at their bids
            p_sell_etf = q["bid"] - basket_ask    # sell ETF, buy stocks at their asks, create ETF
            profit = max(p_buy_etf, p_sell_etf)
            if profit <= c["ETF_ARB_THRESHOLD"]:
                continue
            if self.rate.used() > c["RATE_BUDGET_PER_MIN"] - 40:
                return  # keep budget for Strategy A and manual trades
            if p_buy_etf >= p_sell_etf:
                etf_side, stock_side, conv = "buy", "sell", "redeem"
            else:
                etf_side, stock_side, conv = "sell", "buy", "create"
            units = min(c["ETF_ARB_UNITS"], self.room(etf, etf_side),
                        *(self.room(k, stock_side) // w for k, w in basket.items()))
            units = int(units * self.size_scale) if self.size_scale < 1 else int(units)
            if units < 1:
                continue
            # cross by up to ETF_ARB_CROSS_SHARE of the apparent profit so a slightly moved quote still fills
            give = c["ETF_ARB_CROSS_SHARE"] * profit
            px = q["ask"] + give if etf_side == "buy" else q["bid"] - give
            log.info("B %s: ETF %.2f/%.2f basket %.2f/%.2f -> %s %d ETF @ %.2f then %s (apparent profit %.2f/unit)", etf,
                     q["bid"], q["ask"], basket_bid, basket_ask, etf_side, units, px, conv, profit)
            if self.send(etf, etf_side, units, px, "B", reserve=20):
                self.pending_conv.append({"etf": etf, "conv": conv, "units": units, "ts": now, "stock_side": stock_side,
                                          "pos0": self.pos.get(etf, 0)})
                self.arb_stamps.append(now)

    def process_conversions(self):
        """Second half of an arb: when the ETF leg has filled, convert and send the stock legs."""
        now = self.clock(); c = self.cfg
        for pc in list(self.pending_conv):
            etf = pc["etf"]; pos = self.pos.get(etf, 0)
            filled = (pos - pc["pos0"]) if pc["conv"] == "redeem" else (pc["pos0"] - pos)
            n = min(pc["units"], filled)
            if n >= 1:
                fn = self.ex.etf_redeem if pc["conv"] == "redeem" else self.ex.etf_create
                kw = {"tag": f"B|r{self.round}/conv|{etf}"} if self.dry_run else {}
                if self.action(fn, etf, n, **kw) is None:
                    continue  # rate limited or failed: try again next tick
                sign = -1 if pc["conv"] == "redeem" else 1
                self.pos[etf] = pos + sign * n
                for k, w in ETFS[etf].items():
                    self.pos[k] = self.pos.get(k, 0) - sign * w * n
                log.info("B %s %s %d units -> hedging with stock %ss", pc["conv"].upper(), etf, n, pc["stock_side"])
                for k, w in ETFS[etf].items():
                    lq = self.live_quote(k)
                    if lq is None:
                        continue
                    px = lq["ask"] * (1 + 3 * c["CROSS_FRAC"]) if pc["stock_side"] == "buy" else lq["bid"] * (1 - 3 * c["CROSS_FRAC"])
                    self.send(k, pc["stock_side"], w * n, px, "B", reserve=10)
                pc["units"] -= n; pc["pos0"] = self.pos[etf]
                if pc["units"] <= 0:
                    self.pending_conv.remove(pc)
            elif now - pc["ts"] > c["ORDER_TTL_SECS"] + 2 * c["POLL_SECS"]:
                log.info("B %s: ETF leg did not fill; dropping the arb", etf)
                self.pending_conv.remove(pc)

    def strategy_c(self, adv: dict):
        c = self.cfg
        if not c["STRAT_C"] or self.c_done or self.direct_streak < c["STRAT_C_MIN_CONSECUTIVE"]:
            return
        num = adv.get("number")
        if num is None or num > c["STRAT_C_MAX_ADVISORY"]:
            return
        prec = (MODEL["advisory"]["official"].get(str(num)) or {}).get("direct_precision") or 0.0
        if prec < c["STRAT_C_MIN_PRECISION"]:
            return
        self.c_done = True
        frac = c["STRAT_C_SIZE"] * self.size_scale
        insurer = "REIN" if self.loss >= HAVEN_TOP else "HAVEN"
        log.info("C: %d consecutive direct calls at advisory %s (precision %.2f) -> short PORT, long RECON, short %s at %.0f%% of limits",
                 self.direct_streak, num, prec, insurer, 100 * frac)
        for sym, side in (("PORT", "sell"), ("RECON", "buy"), (insurer, "sell")):
            q = self.live_quote(sym)
            if q is None:
                continue
            target = int(LIMITS[sym] * frac); pos = self.pos.get(sym, 0)
            want = (target - pos) if side == "buy" else (target + pos)
            want -= self.resting(sym, side)
            if want > 0:
                px = q["ask"] * (1 + c["CROSS_FRAC"]) if side == "buy" else q["bid"] * (1 - c["CROSS_FRAC"])
                self.send(sym, side, want, px, "C")

    # ------------------------------------------------------------------ periodic
    def tick(self):
        now = self.clock(); c = self.cfg
        try:
            if now - self.last["expire"] >= 1.0:
                self.last["expire"] = now; self.expire_orders()
            if now - self.last["poll"] >= c["POLL_SECS"]:
                self.last["poll"] = now; self.poll()
            if self.pending_conv:
                self.process_conversions()
            if now - self.last["recheck"] >= c["RECHECK_SECS"] and self.fair:
                self.last["recheck"] = now
                self.compute_fairs(); self.strategy_a("recheck")
            if now - self.last["arb"] >= c["ETF_ARB_EVERY_SECS"]:
                self.last["arb"] = now; self.strategy_b()
            if now - self.last["pnl"] >= c["PNL_EVERY_SECS"]:
                self.last["pnl"] = now; self.report_pnl()
            if now - self.last["state"] >= 2.0:
                self.last["state"] = now; self.write_state()
        except Exception:
            log.exception("tick failed")

    def pnl_estimate(self):
        mtm = 0.0
        for sym, p in self.pos.items():
            m = self.mid(sym)
            if p and m is not None:
                mtm += p * m
        return self.cash_est + mtm

    def report_pnl(self):
        pos = " ".join(f"{k}:{v:+d}" for k, v in self.pos.items() if v)
        log.info("PNL ~%.0f | round %s %s | storms left %d | open orders %d | actions/min %d | pos %s | msgs %s",
                 self.pnl_estimate(), self.round, self.phase, self.storms_remaining, len(self.orders), self.rate.used(),
                 pos or "flat", dict(self.counts))

    def write_state(self):
        if self.dry_run:
            return
        try:
            st = {"written_at": time.time(), "round": self.round, "phase": self.phase, "storm": self.storm,
                  "storms_remaining": self.storms_remaining, "cum": self.cum, "loss_final": self.loss, "prelim": self.prelim,
                  "prelim_loss": self.prelim_loss, "fair": self.fair, "sigma": self.sigma, "sigma_future": self.sigma_future,
                  "quotes": {k: {"bid": v["bid"], "ask": v["ask"]} for k, v in self.quotes.items()}, "positions": self.pos,
                  "open_orders": len(self.orders), "pnl_estimate": self.pnl_estimate(), "connected": self.connected}
            tmp = HERE / (self.cfg["STATE_FILE"] + ".tmp")
            tmp.write_text(json.dumps(st))
            tmp.replace(HERE / self.cfg["STATE_FILE"])
        except Exception as exc:
            log.debug("state write failed: %s", exc)


# ======================================================================================================
def run_live(cfg: dict, size_scale: float):
    key = load_key()
    import websockets
    from websockets.sync.client import connect

    ex = Exchange(key, cfg["BASE_URL"])
    bot = Bot(ex, time.monotonic, cfg, dry_run=False, size_scale=size_scale)
    try:
        log.info("exchange clock: %s", ex.time())
    except ExchangeError as exc:
        log.error("cannot reach the exchange: %s (check the key / network). Continuing; will retry.", exc)
    bot.poll()
    frames: queue.Queue = queue.Queue()
    ws_url = cfg["WS_URL"] or cfg["BASE_URL"].replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/ws/v1/market-data"
    subscribe = {"action": "subscribe", "symbols": SYMBOLS, "channels": ["news", "quotes"]}

    def reader():
        backoff = 1.0
        while True:
            try:
                with connect(ws_url, additional_headers={"X-API-Key": key, "User-Agent": "ctc-hurricane-bot/1.0"},
                             open_timeout=10, close_timeout=2) as ws:
                    ws.send(json.dumps(subscribe))
                    frames.put({"type": "_connected"})
                    log.info("websocket connected and subscribed")
                    backoff = 1.0
                    for raw in ws:
                        try:
                            frames.put(json.loads(raw))
                        except ValueError:
                            log.debug("bad frame: %r", raw[:200])
            except Exception as exc:  # OSError, websockets errors, anything: reconnect
                frames.put({"type": "_disconnected"})
                log.warning("websocket disconnected (%s); reconnecting in %.0fs", exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    threading.Thread(target=reader, daemon=True, name="ws").start()
    log.info("bot running (size scale %.2f). Strategies: A=%s B=%s C=%s. Ctrl-C to stop.", size_scale, cfg["STRAT_A"], cfg["STRAT_B"], cfg["STRAT_C"])
    while True:
        try:
            try:
                frame = frames.get(timeout=0.2)
                bot.handle_frame(frame)
                while True:  # drain what is queued so bursts are handled at once
                    frame = frames.get_nowait()
                    bot.handle_frame(frame)
            except queue.Empty:
                pass
            bot.tick()
        except KeyboardInterrupt:
            log.info("stopping; open orders are left as they are (use manual.py 'cancel all' to clear)")
            return
        except Exception:
            log.exception("main loop error (continuing)")
            time.sleep(0.5)


# ======================================================================================================
def run_replay(cfg: dict, season: int, size_scale: float, quiet: bool = False):
    folder = HERE / "history" / f"season-{season:03d}"
    feed = [json.loads(line) for line in open(folder / "feed.jsonl")]
    with open(folder / "market.csv") as f:
        rows = list(csv.DictReader(f))
    settlement = json.load(open(folder / "settlement.json"))
    opens = {}
    for r in rows:
        opens.setdefault(int(r["round"]), r["time"])

    def to_real(rnd: int, t: str) -> float:
        h, mn, s = (int(x) for x in t[11:19].split(":"))
        oh, om, os_ = (int(x) for x in opens[rnd][11:19].split(":"))
        return (rnd - 1) * ROUND_SECS + ((h - oh) * 3600 + (mn - om) * 60 + (s - os_)) / 30.0

    events = []
    for i, r in enumerate(rows):
        events.append((int(r["round"]), r["time"], 0, i, r))
    for m in feed:
        events.append((int(m["round"]), m["time"], 1, m["seq"], m))
    events.sort(key=lambda e: (e[0], e[1], e[2], e[3]))

    clock_now = [0.0]
    rex = ReplayExchange(cfg["DRY_RUN_SPREAD_FRAC"], cfg["DRY_RUN_LATENCY_SAMPLES"], cfg["TICK"])
    bot = Bot(rex, lambda: clock_now[0], cfg, dry_run=True, size_scale=size_scale)
    if quiet:
        logging.getLogger().handlers[1].setLevel(logging.WARNING)
    log.info("===== DRY RUN season %03d: %s =====", season, folder)
    last_round = 0
    for rnd, t, kind, key, payload in events:
        clock_now[0] = to_real(rnd, t)
        if rnd != last_round:
            last_round = rnd
            bot.round = rnd; bot.phase = "pre_storm"
            log.info("--- round %d opens at %s (real t=%.0fs) ---", rnd, t, clock_now[0])
        if kind == 0:
            rex.on_sample(key, payload)  # key = row index
            for s in SYMBOLS:
                bid, ask, _m = rex.quotes[s]
                bot.on_quote({"symbol": s, "bid": bid, "ask": ask})
        else:
            bot.handle_frame({"type": "news", "message": payload})
        bot.tick()
    # settle
    bot.poll()
    final_pnl = rex.cash + sum(rex.pos[s] * settlement[s] for s in SYMBOLS)
    by_tag = {tag: rex.tag_cash[tag] + sum(p * settlement[s] for s, p in rex.tag_pos[tag].items()) for tag in set(rex.tag_cash) | set(rex.tag_pos)}
    by_strat = collections.defaultdict(float)
    for tag, v in by_tag.items():
        by_strat[tag[0]] += v
    log.warning("DRY RUN season %03d done: PnL at settlement = %.0f  (by strategy: %s; fills %d, order actions %d, final pos %s)",
                season, final_pnl, {k: round(v) for k, v in by_strat.items()}, len(rex.fills), rex.actions,
                {k: v for k, v in rex.pos.items() if v})
    return final_pnl, by_tag, rex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", metavar="N[-M]", help="replay research season N (or seasons N-M) instead of connecting")
    ap.add_argument("--size", type=float, default=1.0, help="scale all position sizes by this factor (e.g. 0.2 on the testnet)")
    ap.add_argument("--quiet", action="store_true", help="dry run: only print the summary lines")
    ap.add_argument("--breakdown", action="store_true", help="dry run over several seasons: PnL by round/stage/symbol")
    ap.add_argument("--latency", type=int, help="dry run: order latency in 2-second samples (default %(default)s)", default=CONFIG["DRY_RUN_LATENCY_SAMPLES"])
    ap.add_argument("--no-b", action="store_true", help="turn Strategy B off")
    ap.add_argument("--url", help="exchange base URL (default %(default)s)", default=CONFIG["BASE_URL"])
    ap.add_argument("--ws-url", help="websocket URL (default: derived from --url)")
    ap.add_argument("--c", action="store_true", help="turn Strategy C on")
    ap.add_argument("--spread", type=float, help="dry run: synthetic full spread fraction (default %(default)s)", default=CONFIG["DRY_RUN_SPREAD_FRAC"])
    args = ap.parse_args()
    cfg = dict(CONFIG)
    cfg["BASE_URL"] = args.url
    if args.ws_url:
        cfg["WS_URL"] = args.ws_url
    if args.no_b:
        cfg["STRAT_B"] = False
    if args.c:
        cfg["STRAT_C"] = True
    cfg["DRY_RUN_SPREAD_FRAC"] = args.spread
    cfg["DRY_RUN_LATENCY_SAMPLES"] = args.latency
    if args.dry_run:
        setup_logging(HERE / "dryrun.log")
        lo, _, hi = args.dry_run.partition("-")
        seasons = range(int(lo), int(hi or lo) + 1)
        results = []
        for s in seasons:
            pnl, by_tag, _ = run_replay(cfg, s, args.size, quiet=args.quiet or len(seasons) > 1)
            results.append((s, pnl, by_tag))
        if len(results) > 1:
            import numpy as np
            p = np.array([r[1] for r in results])
            letters = sorted({t[0] for r in results for t in r[2]})
            print(f"\n{len(results)} seasons: mean PnL {p.mean():.0f}  median {np.median(p):.0f}  std {p.std():.0f}  min {p.min():.0f}  max {p.max():.0f}  "
                  f"share positive {np.mean(p > 0):.2f}")
            for t in letters:
                v = np.array([sum(val for tag, val in r[2].items() if tag[0] == t) for r in results])
                print(f"  strategy {t}: mean {v.mean():.0f}  median {np.median(v):.0f}  std {v.std():.0f}  min {v.min():.0f}  share positive {np.mean(v > 0):.2f}")
            if args.breakdown:
                print("\nbreakdown by strategy|round/stage|symbol (total over seasons, seasons with trades, mean per season when traded, worst season):")
                detail = collections.defaultdict(list)
                for r in results:
                    for tag, val in r[2].items():
                        detail[tag].append(val)
                for tag, vals in sorted(detail.items(), key=lambda kv: sum(kv[1])):
                    v = np.array(vals)
                    print(f"  {tag:24} total {v.sum():8.0f}  n {len(v):3d}  mean {v.mean():7.0f}  worst {v.min():7.0f}")
        return
    if "CTC_API_KEY" not in __import__("os").environ:
        load_key()  # exits with the clear message
    setup_logging(HERE / CONFIG["LOG_FILE"])
    run_live(cfg, args.size)


if __name__ == "__main__":
    main()
