#!/usr/bin/env python3
"""Manual override CLI. Run alongside bot.py (same key is fine; it shares the TEAM rate limit of 180/min).

    export CTC_API_KEY=...
    python3 manual.py

Commands:
    buy  PORT 50 71.2        limit buy 50 PORT at 71.20
    buy  PORT 50             buy 50 PORT crossing the spread (limit at best ask + 0.2%)
    sell HAVEN 100 [price]   same for sells
    flat GRID                send one crossing order to take GRID to zero
    flat all                 flatten every symbol
    cancel all | cancel PORT | cancel <order_id>
    pos                      positions
    orders                   open orders
    book PORT                order book
    fair                     fair values from bot_state.json (written by bot.py every few seconds) + positions
    fair PORT +5             bot adds 5.00 to PORT's fair value from now on (announcement handling)
    fair PORT 60             bot uses 60.00 as PORT's fair value
    fair PORT skip           bot stops trading PORT (Strategy A)
    fair PORT clear          remove PORT's override;  fair clear  removes all
    create ARCH 2 / redeem REBLD 1
    time                     competition clock
    quit
"""
from __future__ import annotations

import json
import shlex
import sys
import time
from pathlib import Path

from exchange import Exchange, ExchangeError, load_key

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / "bot_state.json"
OVERRIDES_FILE = HERE / "overrides.json"


def set_override(args):
    """fair SYM +5 | SYM 60 | SYM skip | SYM clear | clear"""
    ov = json.loads(OVERRIDES_FILE.read_text()) if OVERRIDES_FILE.exists() and OVERRIDES_FILE.stat().st_size else {}
    if args[0].lower() == "clear":
        ov = {}
    else:
        sym = args[0].upper(); what = args[1].lower() if len(args) > 1 else ""
        if what == "clear":
            ov.pop(sym, None)
        elif what == "skip":
            ov[sym] = {"skip": True}
        elif what.startswith(("+", "-")):
            ov[sym] = {"add": float(what)}
        else:
            ov[sym] = {"set": float(what)}
    OVERRIDES_FILE.write_text(json.dumps(ov))
    print("overrides now:", ov or "none", "(bot picks this up within 2 s; check its log for 'OVERRIDES loaded')")
CROSS_FRAC = 0.002


def best_quote(ex: Exchange, sym: str):
    """(bid, ask) from GET /orderbook; tolerant to shape."""
    book = ex.orderbook(sym)
    bids = book.get("bids") or book.get("bid") or []
    asks = book.get("asks") or book.get("ask") or []

    def px(level):
        if isinstance(level, dict):
            return float(level.get("price"))
        return float(level[0])

    bid = max((px(l) for l in bids), default=None) if isinstance(bids, list) else None
    ask = min((px(l) for l in asks), default=None) if isinstance(asks, list) else None
    if bid is None and isinstance(book.get("best_bid"), (int, float)):
        bid = float(book["best_bid"])
    if ask is None and isinstance(book.get("best_ask"), (int, float)):
        ask = float(book["best_ask"])
    return bid, ask


def crossing_price(ex: Exchange, sym: str, side: str) -> float:
    bid, ask = best_quote(ex, sym)
    if side == "buy":
        if ask is None:
            raise RuntimeError(f"no ask for {sym}; give a price")
        return round(ask * (1 + CROSS_FRAC), 2)
    if bid is None:
        raise RuntimeError(f"no bid for {sym}; give a price")
    return round(bid * (1 - CROSS_FRAC), 2)


def show_positions(ex: Exchange):
    pos = ex.parse_positions(ex.positions())
    if not pos:
        print("  (flat)")
    for k, v in sorted(pos.items()):
        if v:
            print(f"  {k:6} {v:+6d}")
    return pos


def show_fair(ex: Exchange):
    if STATE_FILE.exists():
        st = json.loads(STATE_FILE.read_text())
        age = time.time() - st.get("written_at", 0)
        print(f"bot state (written {age:.0f}s ago): round {st.get('round')} phase {st.get('phase')} storms_remaining {st.get('storms_remaining')}"
              f" loss_final {st.get('loss_final')} prelim pending: {st.get('prelim') is not None}")
        print(f"  {'sym':6} {'fair':>8} {'sigma':>6} {'bid':>8} {'ask':>8} {'pos':>6}")
        q = st.get("quotes", {})
        for k, f in st.get("fair", {}).items():
            qq = q.get(k) or {}
            print(f"  {k:6} {f:8.2f} {st.get('sigma', {}).get(k, 0):6.2f} {qq.get('bid') or float('nan'):8.2f} {qq.get('ask') or float('nan'):8.2f} {st.get('positions', {}).get(k, 0):6d}")
    else:
        print("no bot_state.json yet (bot.py writes it); live positions:")
    print("positions from the exchange:")
    show_positions(ex)


def main():
    ex = Exchange(load_key())
    print(__doc__)
    while True:
        try:
            line = input("manual> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        try:
            args = shlex.split(line)
            cmd = args[0].lower()
            if cmd in ("quit", "exit", "q"):
                return
            elif cmd in ("buy", "sell"):
                sym = args[1].upper(); qty = int(args[2])
                price = float(args[3]) if len(args) > 3 else crossing_price(ex, sym, cmd)
                print(" ->", ex.place_order(sym, cmd, qty, price))
            elif cmd == "flat":
                pos = ex.parse_positions(ex.positions())
                syms = [a.upper() for a in args[1:]] if args[1:] and args[1].lower() != "all" else [k for k, v in pos.items() if v]
                for sym in syms:
                    q = pos.get(sym, 0)
                    if not q:
                        print(f"  {sym}: already flat"); continue
                    side = "sell" if q > 0 else "buy"
                    try:
                        ex.cancel_all(sym)
                    except ExchangeError as exc:
                        print("  cancel_all:", exc)
                    print(f"  {sym}: {side} {abs(q)} ->", ex.place_order(sym, side, abs(q), crossing_price(ex, sym, side)))
            elif cmd == "cancel":
                what = args[1] if len(args) > 1 else "all"
                if what.lower() == "all":
                    print(" ->", ex.cancel_all())
                elif what.isupper() and len(what) <= 6:
                    print(" ->", ex.cancel_all(what))
                else:
                    print(" ->", ex.cancel(what))
            elif cmd in ("pos", "positions"):
                show_positions(ex)
            elif cmd == "orders":
                for o in ex.parse_open_orders(ex.open_orders()):
                    print(f"  {o['id']} {o['symbol']:6} {o['side']:4} {o['quantity']:5d} @ {o['price']}")
            elif cmd == "book":
                print(json.dumps(ex.orderbook(args[1].upper()), indent=1)[:2000])
            elif cmd == "fair":
                if len(args) > 1:
                    set_override(args[1:])
                else:
                    show_fair(ex)
            elif cmd in ("create", "redeem"):
                sym = args[1].upper(); qty = int(args[2]) if len(args) > 2 else 1
                print(" ->", (ex.etf_create if cmd == "create" else ex.etf_redeem)(sym, qty))
            elif cmd == "time":
                print(ex.time())
            else:
                print("unknown command; see the list above")
        except ExchangeError as exc:
            print("exchange error:", exc)
        except (IndexError, ValueError) as exc:
            print("bad arguments:", exc)
        except Exception as exc:  # keep the CLI alive no matter what
            print("error:", type(exc).__name__, exc)


if __name__ == "__main__":
    sys.exit(main())
