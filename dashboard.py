#!/usr/bin/env python3
"""Live dashboard: PnL, positions, fills, open orders, the bot's order log and fair-vs-quote, refreshed every 2 s.

    python3 dashboard.py            # then open http://localhost:8787
    python3 dashboard.py --port 9000

Reads the exchange (GET /trades, /positions, /orders/open, /competitions/time) with the same key as the bot, plus
bot_state.json and bot.log written by bot.py. Stdlib only (plus requests via exchange.py).
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from exchange import Exchange, load_key

HERE = Path(__file__).resolve().parent
ORDER_RE = re.compile(r"^(\d\d:\d\d:\d\d) \w+ (ORDER|fill |FILL|CANCEL|B |A/|REPORT|ADVISORY|IMPACT|NEWS|!!!|OVERRIDES|RATE|REJECTED|ORDER REJECTED|competition state|NEW SEASON|===)")

state = {"data": {}, "lock": threading.Lock()}

TRADES_FILE = HERE / "trades_seen.jsonl"
_trades_seen: dict = {}


def merged_trades(live: list) -> list:
    """Union of every trade record seen so far (persisted) and the live list; newest first."""
    global _trades_seen
    if not _trades_seen and TRADES_FILE.exists():
        for line in TRADES_FILE.read_text().splitlines():
            try:
                t = json.loads(line); _trades_seen[t["trade_id"]] = t
            except Exception:
                pass
    new = [t for t in live if t.get("trade_id") and t["trade_id"] not in _trades_seen]
    if new:
        with TRADES_FILE.open("a") as f:
            for t in new:
                _trades_seen[t["trade_id"]] = t; f.write(json.dumps(t) + "\n")
    return sorted(_trades_seen.values(), key=lambda t: t.get("executed_at", ""), reverse=True)

BOT_ORDER_RE = re.compile(r"^(\d\d:\d\d:\d\d) INFO ORDER (\w+)\s+(buy|sell)\s+(\d+) @ ([\d.]+) \[(\w)\]")
BOT_CONV_RE = re.compile(r"^(\d\d:\d\d:\d\d) INFO B (REDEEM|CREATE) (\w+) (\d+) units")
BASKETS = {k: v["basket"] for k, v in json.loads((HERE / "case.json").read_text())["etfs"].items()}


def tsec(s):
    h, m, sec = s.split(":"); return int(h) * 3600 + int(m) * 60 + float(sec)


def bot_attribution(trades, lines):
    """Match exchange fills to the bot's own ORDER / REDEEM / CREATE log lines. Returns (bot_trades, bot_pos, cash)."""
    orders = []; convs = []
    for l in lines:
        m = BOT_ORDER_RE.match(l)
        if m:
            orders.append([tsec(m.group(1)), m.group(2), m.group(3), int(m.group(4)), float(m.group(5)), m.group(6), 0])
            continue
        m = BOT_CONV_RE.match(l)
        if m:
            convs.append([tsec(m.group(1)), m.group(2), m.group(3), int(m.group(4)), False])
    bot = []; pos = {}; cash = 0.0
    for t in sorted(trades, key=lambda t: t.get("executed_at", "")):
        ts = tsec(t["executed_at"][11:19]); q = int(t["quantity"]); sign = 1 if str(t.get("side", "")).startswith("s") else -1
        kind = t.get("kind", "trade")
        if kind == "trade":
            hit = None
            for o in orders:
                if o[1] == t["symbol"] and o[2] == t["side"] and -1 <= ts - o[0] <= 50 and o[6] + q <= o[3]:
                    hit = o; break
            if hit:
                hit[6] += q; bot.append(t); cash += sign * q * float(t["price"])
                pos[t["symbol"]] = pos.get(t["symbol"], 0) - sign * q
        else:  # etf_create / etf_redeem: the ETF side is the record; apply the basket move if the bot logged this conversion
            for c in convs:
                if not c[4] and c[2] == t["symbol"] and abs(ts - c[0]) <= 8 and c[3] == q:
                    c[4] = True; bot.append(t)
                    d = 1 if c[1] == "CREATE" else -1
                    pos[t["symbol"]] = pos.get(t["symbol"], 0) + d * q
                    for k, w in BASKETS[t["symbol"]].items():
                        pos[k] = pos.get(k, 0) - d * w * q
                    break
    return bot, {k: v for k, v in pos.items() if v}, cash


def pnl_from(trades, positions):
    cash = 0.0; n = 0
    for t in trades:
        if t.get("kind", "trade") != "trade":
            continue
        q = float(t["quantity"]); p = float(t["price"])
        cash += q * p if str(t.get("side", "")).startswith("s") else -q * p
        n += 1
    mtm = sum(float(p["quantity"]) * float(p.get("current_price") or 0) for p in positions)
    return cash, mtm, n


def poll_loop(ex: Exchange, interval: float):
    while True:
        d = {"time": time.strftime("%H:%M:%S")}
        try:
            d["clock"] = ex.time()
        except Exception as exc:
            d["clock"] = {"error": str(exc)[:120]}
        try:
            d["positions"] = ex.positions().get("positions", [])
        except Exception as exc:
            d["positions"] = []; d["pos_error"] = str(exc)[:120]
        try:
            d["trades"] = merged_trades(ex._req("GET", "/trades?limit=500").get("trades", []))
        except Exception as exc:
            d["trades"] = []; d["trades_error"] = str(exc)[:120]
        try:
            d["open_orders"] = ex.parse_open_orders(ex.open_orders())
        except Exception as exc:
            d["open_orders"] = []
        cash, mtm, n = pnl_from(d["trades"], d["positions"])
        d["pnl"] = {"total": cash + mtm, "cash": cash, "mtm": mtm, "fills": n}
        try:
            st = json.loads((HERE / "bot_state.json").read_text())
            st["age"] = time.time() - st.get("written_at", 0)
            d["bot"] = st
        except Exception:
            st = None; d["bot"] = None
        try:
            all_lines = (HERE / "bot.log").read_text(errors="replace").splitlines()
            d["log"] = [l[:220] for l in all_lines[-400:] if ORDER_RE.match(l)][-40:]
        except Exception:
            all_lines = []; d["log"] = []
        # --- the bot's own slice: fills matched to its order log, net positions, PnL at marks
        marks = {p["symbol"]: float(p.get("current_price") or 0) for p in d["positions"]}
        if st:
            for s2, q2 in (st.get("quotes") or {}).items():
                if s2 not in marks and q2.get("bid") is not None and q2.get("ask") is not None:
                    marks[s2] = 0.5 * (q2["bid"] + q2["ask"])
        try:
            bt, bpos, bcash = bot_attribution(d["trades"], all_lines)
        except Exception as exc:
            bt, bpos, bcash = [], {}, 0.0; d["attr_error"] = str(exc)[:200]
        bmtm = sum(q * marks.get(s2, 0.0) for s2, q in bpos.items())
        d["bot_trades"] = list(reversed(bt))
        d["bot_pos"] = [{"symbol": s2, "quantity": q, "mark": marks.get(s2)} for s2, q in sorted(bpos.items())]
        d["bot_pnl"] = {"total": bcash + bmtm, "cash": bcash, "mtm": bmtm, "fills": sum(1 for t in bt if t.get("kind", "trade") == "trade")}
        # bot's open orders: those matching a recent ORDER line by symbol/side/price
        recent = [BOT_ORDER_RE.match(l) for l in all_lines[-300:]]
        keys = {(m.group(2), m.group(3), round(float(m.group(5)), 2)) for m in recent if m}
        d["bot_orders"] = [o for o in d["open_orders"] if (o["symbol"], o["side"], round(float(o["price"] or 0), 2)) in keys]
        with state["lock"]:
            state["data"] = d
        time.sleep(interval)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Hurricane bot</title>
<style>
body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;background:#111;color:#ddd}
header{display:flex;gap:28px;align-items:baseline;padding:12px 18px;background:#1b1b1b;border-bottom:1px solid #333;flex-wrap:wrap}
h1{font-size:16px;margin:0;color:#fff}.big{font-size:34px;font-weight:700}.pos{color:#4cd964}.neg{color:#ff5b5b}
.muted{color:#888;font-size:12px}main{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px 18px}
section{background:#1b1b1b;border:1px solid #333;border-radius:8px;padding:10px 12px;min-height:60px}
h2{font-size:13px;margin:0 0 8px;color:#aaa;text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
td,th{padding:3px 6px;text-align:right;border-bottom:1px solid #2a2a2a}th{color:#888;font-weight:500}
td:first-child,th:first-child{text-align:left}.wide{grid-column:1/-1}
pre{font-size:12px;line-height:1.35;margin:0;white-space:pre-wrap;color:#ccc;max-height:360px;overflow:auto}
.new{background:#263a26}.buy{color:#4cd964}.sell{color:#ff8a65}.warn{color:#ffcc00}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Hurricane bot</h1><div><span class="muted">bot PnL</span> <span id="pnl" class="big">…</span></div>
<div><span class="muted">bot fills</span> <b id="fills">…</b></div><div><span class="muted">team total</span> <b id="team">…</b></div><div><span class="muted">exchange</span> <b id="clock">…</b></div>
<div><span class="muted">bot</span> <b id="botst">…</b></div><div class="muted" id="upd"></div></header>
<main>
<section><h2>Bot positions (net of conversions)</h2><table id="postab"></table></section>
<section><h2>Bot open orders</h2><table id="ordtab"></table></section>
<section class="wide"><h2>Bot fills (newest first, green = new since last refresh)</h2><table id="trtab"></table></section>
<section><h2>Fair value vs quote (bot)</h2><table id="fairtab"></table></section>
<section><h2>Bot log</h2><pre id="log"></pre></section>
</main>
<script>
let seen=new Set(), first=true;
function cls(v){return v>0?'pos':v<0?'neg':''}
function fmt(v,d=2){return v==null?'':Number(v).toFixed(d)}
async function tick(){
  let d; try{ d=await (await fetch('/data')).json(); }catch(e){ document.getElementById('upd').textContent='dashboard server unreachable'; return; }
  const p=d.bot_pnl||{}; const el=document.getElementById('pnl'); el.textContent=(p.total>=0?'+':'')+fmt(p.total,0); el.className='big '+cls(p.total);
  document.getElementById('fills').textContent=p.fills;
  const tp=d.pnl||{}; const te=document.getElementById('team'); te.textContent=(tp.total>=0?'+':'')+fmt(tp.total,0); te.className=cls(tp.total);
  const c=d.clock||{}; document.getElementById('clock').textContent=(c.state||c.error||'?')+' '+(c.fictional_time||'').slice(11,19);
  const b=d.bot; document.getElementById('botst').textContent=b?`round ${b.round} ${b.phase} · ${b.connected?'connected':'DISCONNECTED'} · state ${fmt(b.age,0)}s old`:'no bot_state.json';
  document.getElementById('botst').className=b&&b.connected&&b.age<10?'':'warn';
  document.getElementById('upd').textContent='updated '+d.time;
  let h='<tr><th>sym</th><th>qty</th><th>mark</th><th>value</th></tr>';
  for(const q of d.bot_pos||[]) h+=`<tr><td>${q.symbol}</td><td>${q.quantity}</td><td>${fmt(q.mark)}</td><td>${fmt(q.quantity*(q.mark||0),0)}</td></tr>`;
  if(!(d.bot_pos||[]).length) h+='<tr><td colspan=4 class="muted">flat (every arb hedged)</td></tr>';
  document.getElementById('postab').innerHTML=h;
  h='<tr><th>sym</th><th>side</th><th>qty</th><th>price</th><th>id</th></tr>';
  for(const o of d.bot_orders||[]) h+=`<tr><td>${o.symbol}</td><td class="${o.side}">${o.side}</td><td>${o.quantity}</td><td>${fmt(o.price)}</td><td class="muted">${String(o.id).slice(0,8)}</td></tr>`;
  if(!(d.bot_orders||[]).length) h+='<tr><td colspan=5 class="muted">none</td></tr>';
  document.getElementById('ordtab').innerHTML=h;
  h='<tr><th>time</th><th>sym</th><th>side</th><th>qty</th><th>price</th><th>value</th><th>kind</th></tr>';
  for(const t of (d.bot_trades||[]).slice(0,80)){ const isNew=!first&&!seen.has(t.trade_id); seen.add(t.trade_id);
    h+=`<tr class="${isNew?'new':''}"><td>${(t.executed_at||'').slice(11,19)}</td><td>${t.symbol}</td><td class="${t.side}">${t.side}</td><td>${t.quantity}</td><td>${fmt(t.price)}</td><td>${fmt(t.quantity*t.price,0)}</td><td class="muted">${t.kind||''}</td></tr>`; }
  first=false; document.getElementById('trtab').innerHTML=h;
  h='<tr><th>sym</th><th>bid</th><th>ask</th><th>fair</th><th>edge</th><th>pos</th></tr>';
  if(b&&b.fair){ for(const s of Object.keys(b.fair)){ const q=(b.quotes||{})[s]||{}; const f=b.fair[s]; const mid=(q.bid!=null&&q.ask!=null)?(q.bid+q.ask)/2:null; const e=mid!=null?f-mid:null;
    h+=`<tr><td>${s}</td><td>${fmt(q.bid)}</td><td>${fmt(q.ask)}</td><td>${fmt(f)}</td><td class="${cls(e)}">${e==null?'':(e>0?'+':'')+fmt(e)}</td><td>${(b.positions||{})[s]||0}</td></tr>`; } }
  document.getElementById('fairtab').innerHTML=h;
  document.getElementById('log').textContent=(d.log||[]).slice().reverse().join('\\n');
}
tick(); setInterval(tick,2000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/data"):
            with state["lock"]:
                body = json.dumps(state["data"]).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode(); ctype = "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()
    ex = Exchange(load_key())
    threading.Thread(target=poll_loop, args=(ex, args.interval), daemon=True).start()
    print(f"dashboard at http://localhost:{args.port}  (Ctrl-C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
