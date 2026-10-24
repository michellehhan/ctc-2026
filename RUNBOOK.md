# RUNBOOK — Hurricane Season bot

## Files

| File | What |
|---|---|
| `research.py` | Stage 1. Fits the settlement model on the 600 seasons, prints the tables, writes `model.json`. Re-run only if the data changes (~10 s). |
| `model.json` | Fitted parameters + per-storm table. Loaded by `valuation.py`. |
| `valuation.py` | Stage 2. `fair_values(...)`, `fair_and_sigma(...)`. `python3 valuation.py` validates against the house opens. |
| `exchange.py` | REST client (orders, cancels, positions, ETF create/redeem). Needs `CTC_API_KEY`. |
| `bot.py` | Stage 3. The bot. `--dry-run N` replays `history/season-NNN`. Config block at the top. |
| `manual.py` | Manual override CLI (same key). Reads `bot_state.json` that bot.py writes every 2 s. |
| `bot.log` / `dryrun.log` | Every message count, order, fill, PnL line (live / dry run). |

Dependencies: Python 3.10+, `numpy`, `requests`, `websockets>=14`. Nothing else.

## Start (live / testnet)

```bash
export CTC_API_KEY=your_bot_key      # from the Team page. Missing key -> bot exits with a clear message.
python3 bot.py --size 0.2            # testnet: sizes at 20% of the limits
python3 bot.py                       # competition: full sizes
```

In a second terminal: `python3 manual.py`.
To record the feed for the starter-kit check (optional, in a third terminal):

```bash
python3 starter_kit/record_feed.py --url wss://cornelltradingcompetition.org --key $CTC_API_KEY --out recording.jsonl
python3 starter_kit/check_recording.py recording.jsonl history/      # after a season: "recording matches the published feed"
```

Testnet checklist (first run with the real key):
1. Watch the first lines: `exchange clock: ...`, `raw positions payload`, `raw open_orders payload`, `websocket connected and subscribed`.
   If the positions/open-orders payload shape looks different from `[{"symbol","quantity"}]` / `{"orders":[...]}`,
   tell me (the parsers in `exchange.py` accept several shapes but not every shape).
2. Quotes: the PnL line every 10 s shows `'quote': N` growing. If it stays 0, the quote frame format differs from
   `{"type":"quote","symbol","bid","ask"}` — check `bot.log` for `frame:...` counts and tell me.
3. Place a manual order (`buy FERRY 1 <price>`) and confirm it appears in `orders` and fills show in `pos`.
4. Let it run a full season. Compare the final `PNL ~` line with the exchange's own PnL.

## What the bot does

- **Strategy A (core)** — on every damage report and every 2 s after it, fair value per symbol from the model.
  - Round 3: trades the *level* (settlement is known after the final, nearly known after the prelim). This is where
    almost all the PnL comes from: PORT, HAVEN, ARCH, AQUA, ISLE, GRID, RECON. Goes to the full limit after the final.
  - Rounds 1-2: **off by default** (`SIZE_FINAL_R12 = SIZE_PRELIM_R12 = 0`). Research: the house already prices the
    storm track, and it moves only ~25% of our fair-value change on a final report (5% on a prelim), so there is no
    calibrated edge. If the live house is visibly slower than the testnet house, set `SIZE_FINAL_R12 = 0.3` — it then
    trades the house's *reaction* (pre-report mid + 25% of our change) for 60 s after each final report.
- **Strategy B (ETF arb)** — every 2 s, ETF quote vs basket of stock quotes. ETF leg first (limit crosses by 40% of the
  apparent profit); when it fills, redeem/create and send the stock legs. If the ETF leg misses, nothing else is sent.
  10 units per arb, max 6 per minute. In the replay this earns ~1.2k per season and is positive in 96% of seasons;
  the house's ETF price lags its own basket, especially right after reports.
- **Strategy C** — off (`STRAT_C=False`, or `--c` to enable). Replay over 80 seasons: mean +0.2k/season, worst −2.3k,
  triggers in ~1/5 of seasons; the house reacts to advisories within seconds on the testnet. Turn on only if the live house is slow.
- **Announcements** are printed with a bell and `####` banners and never traded — use `manual.py`.
- **Rate limit**: counts its own order actions, stays under 150/min (team limit 180) and leaves 40/min for manual trades.
- **Orders**: limit, crossing the best quote by 0.1%; cancelled after 8 s if unfilled. Limits include resting orders.
- **State**: season resets automatically when the round number goes back to 1 (testnet rolls to the next season).

## manual.py

```
buy PORT 50 71.2      sell HAVEN 100         (no price = cross the spread)
flat GRID             flat all               cancel all | cancel PORT | cancel <id>
fair                  pos | orders | book PORT | time
create ARCH 2         redeem REBLD 1
```
`fair` shows the bot's fair values, sigma, quotes and positions (from `bot_state.json`, with its age in seconds).

Announcement handling (the bot picks these up within 2 s and logs `OVERRIDES loaded`):
```
fair PORT +5       add 5.00 to PORT's fair value (announcement says PORT is worth more)
fair PORT 60       set PORT's fair value to 60.00
fair PORT skip     stop the bot trading PORT
fair PORT clear    remove the override;   fair clear   removes all
```
ETF fairs follow the overridden stocks automatically.

## What to watch

- `PNL ~x | round r phase | storms left n | open orders k | actions/min a | pos ... | msgs {...}` every 10 s.
  `actions/min` near 150 means the rate limiter is skipping orders (`RATE LIMIT` lines).
- `REPORT PRELIMINARY/FINAL round r` lines followed by `A/final SYM: fair .. quote .. edge .. -> buy/sell n`.
  After the round-3 final you should see a burst of orders within the same second. **Speed matters**: in the research
  data the house re-prices within ~4 s of the final report.
- `ORDER REJECTED ...` — read the body; a position-limit rejection means our resting-order accounting drifted
  (the 3-second poll re-syncs it). Repeated rejections: `cancel all` in manual.py.
- `B SYM: ETF leg did not fill` is normal. `B REDEEM/CREATE ... hedging` means an arb went through.
- `!!! ANNOUNCEMENT` banner: read it, decide by hand (`fair` then `buy/sell`).

## If it disconnects

- The bot reconnects on its own with backoff (1 s → 15 s) and re-subscribes. `websocket disconnected (...)` then
  `websocket connected and subscribed`. The exchange does not resend missed messages, but the 30-second snapshot
  carries every damage report of the round, so a missed report is picked up (`REPORT ... (from snapshot)`).
  The storm-exposure term (seconds the storm sat over a company's sites) cannot be rebuilt from a snapshot; its
  effect is small (≤ 0.5 $ on PORT/GRID/FERRY/GEN).
- If it crashes outright: `python3 bot.py` again. It polls positions at startup and carries on; the season state is
  rebuilt from the next snapshot (within 30 s). Open orders from before are shown in `orders` and expire by TTL only
  if the bot knows them (the poll re-learns them); `cancel all` in manual.py if in doubt.
- If REST calls fail with 401/403: wrong or missing key, or the User-Agent header (set in `exchange.py`).
- Stopping: Ctrl-C. Resting orders are left on the book — `cancel all` in manual.py.

## Dry run (what was tested tonight)

```bash
python3 bot.py --dry-run 1              # one season, full log in dryrun.log
python3 bot.py --dry-run 1-80 --quiet --breakdown    # distribution + PnL by round/stage/symbol
python3 bot.py --dry-run 1-40 --quiet --latency 2 --spread 0.008   # slower fills, wider synthetic spread
```
The replay builds house quotes from `market.csv` (mid ± 0.15%, configurable) with 2 s latency and unlimited depth
at the quote, so it is optimistic about size: the live house will not fill 200 PORT at a stale quote. 80 seasons,
default config: mean +2.2k/season, positive in 79/80 (A: +0.9k, never negative; B: +1.3k, 77/80 positive).
