# Hurricane Season bot — technical report

Cornell Trading Competition, Systematic Equities case. Built the night before the competition from the 600
research seasons in `history/`. Companion to `RUNBOOK.md` (operations) and the code itself.

---

## 1. The problem

Twelve symbols (10 stocks, 2 ETFs) trade for three rounds of 13 real minutes. Each round is one hurricane.
At the close of round 3 every stock settles at a value that is a deterministic function of (a) the cumulative
damage each island business took and (b) the season's total insured loss. ETFs settle at basket value.
Score is PnL after settlement, nothing else.

So the task reduces to three questions:

1. What is the settlement function? (Estimated from history.)
2. What is each stock worth *now*, given what has been revealed so far? (Expected settlement.)
3. When is the house's quote far enough from that value to be worth trading, and how much?

The code answers them in that order: `research.py` → `valuation.py` → `bot.py`.

---

## 2. Stage 1 — estimating the settlement function (`research.py`)

**Data.** 600 seasons × 3 storms. For each season: the full message feed, the house price every 2 s, the
final damage report of each storm (damage % per island company, insured loss in $M), and the settlement value
of each symbol. This is real simulator output, not anything we generated.

**Method.** Ordinary least squares (`numpy.linalg.lstsq`) on candidate functional forms, choosing by residual
standard deviation. No machine learning in the "model-fitting-with-hyperparameters" sense; the forms are
simple enough that OLS on transformed features is exact and interpretable, which matters when you need to
trust a number with 40 seconds left in the round.

**Findings, per company type.**

| Symbol | Form found | Residual std ($) |
|---|---|---|
| ISLE | 60.00 − 0.380 · cumulative damage % | 0.003 (exact) |
| AQUA | 35.00 − 0.200 · cumulative damage % | 0.003 (exact) |
| PORT | 100.0 − 0.56 · damage − 0.015 · exposure | 0.33 (0.76 without exposure) |
| GRID | 50.0 − 0.26 · damage − 0.037 · exposure | 0.34 (0.97 without) |
| FERRY | 25.0 − 0.07 · damage − 0.055 · exposure | 0.41 (1.68 without) |
| RECON | 36.4 + 0.18 · √(100·PORT + 50·GRID + 26·FERRY damage) | 0.25 |
| MAT | 25.7 + 0.32 · √(season loss) | 0.08 |
| GEN | 32.1 + 0.53 · √(GRID damage) + 0.058 · GRID exposure | 0.57 |
| HAVEN | 75 − 0.06 · clip(loss − 400, 0, 900) | 0.003 (exact) |
| REIN | 90 − 0.035 · clip(loss − 1300, 0, ~470) | 0.001 (exact; upper bound not observed) |

Three things worth understanding:

- **Exposure.** The case text says the grid loses revenue "as long as the outage continues" and the ferry
  loses fares "each hour a terminal is closed". Damage % alone left residuals of ~1 $. I defined *exposure*
  as the number of real seconds the storm's radius covered each company's site, weighted by the site weight
  from `case.json` and the storm's intensity, summed over every `storm` observation (one every 2 s). That one
  feature cut the residuals by 2–4×. It is observable live, so the bot accumulates it from the feed.
- **Insured loss is a linear function of damage**: loss = 9·PORT + 6·ISLE + 2.5·AQUA + 7·GRID + 1.2·FERRY
  (residual 0.07 $M). So a damage report pins down the loss contribution too.
- **Insurer layers.** Fit by grid-searching the attachment point and layer width and solving the slope by
  OLS. HAVEN's downside is capped at 21; REIN only moves once the season total passes 1300 $M, which happens
  in roughly the top third of seasons.

**Damage compounds on what is left.** Cumulative damage never exceeds 100% in 600 seasons, and a storm's
damage is negatively correlated with prior damage, while damage expressed as a share of the *remaining*
value is uncorrelated with it. So each storm damages what is still standing. This matters for pricing the
future (section 3).

**Other measurements in the research output:**

- *Prelim vs final report:* final ≈ 0.83 · prelim + small intercept. Residual std 8.3 damage points for PORT,
  4–6 for the others, 86 $M for loss. Prelims overstate damage slightly.
- *Advisory accuracy:* official outlooks match the true steering pattern ~70% of the time at every advisory
  number; a "direct strike" call has 76–81% precision. Wire chatter ~50%. Storms called "direct" average 33%
  PORT damage vs 16% overall.
- *House behaviour:* measured the gap between the house price and the truth at offsets around each report.
  The house moves to the correct level within two 2-second samples of the final report (gap std for PORT:
  2.2 before, 0.11 after). Before the report its price already reflects the storm track, so the prelim adds
  little information it did not have.

Everything is saved to `model.json`, including a table of all 1,800 historical storms (damage as share of
remaining value, loss, exposure) for the Monte Carlo.

---

## 3. Stage 2 — expected settlement (`valuation.py`)

`fair_values(cum_damage, season_loss, storms_remaining, exposure, prelim, prelim_loss)` returns the expected
settlement of every symbol. `fair_and_sigma(...)` also returns two uncertainties.

**Known part.** Damage from storms with a final report, plus the bias-corrected preliminary report of the
current storm if its final is still pending, plus the live exposure total.

**Unknown part: Monte Carlo over historical storms.** For `r` storms still to come, draw `r` storms from the
1,800 historical ones (1,500 pre-drawn scenarios, fixed seed, so the function is pure and repeatable).
Apply them in sequence: each drawn storm's raw damage is multiplied by the value still standing, cumulative
damage and loss are updated, exposure is added. Push every scenario through the settlement function from
section 2 and average. The whole call is ~0.2 ms because it is vectorised numpy over the scenario axis.

Why Monte Carlo rather than plugging in mean damage: the settlement function is nonlinear (square roots,
insurer layers), so E[f(X)] ≠ f(E[X]). HAVEN in particular is concave in loss.

**Validation.** Our fair at the round-1 open matches the house's mean opening price to within 0.3 $ on
most names (HAVEN differs by 1.6 — the house opens HAVEN low relative to realised settlements). At the
round-2 and round-3 opens, fair − house has std ~1 $ for PORT, ~0.5 for most others.

**Two sigmas.**

- `sigma`: uncertainty of *our estimate* given what is known — model residual plus, if a prelim is pending,
  the prelim-vs-final spread propagated through the function by a symmetric perturbation.
- `sigma_future`: spread of the settlement across the Monte Carlo scenarios, i.e. what the remaining storms
  can still do. Zero once all three finals are in.

The bot's required edge is `EDGE_K · sigma + FUTURE_K · sigma_future + max(5 cents, 0.2% of price)`.

---

## 4. Stage 3 — the bot (`bot.py`)

### 4.1 Strategy A: trade the stale quote after a report

On every damage report (and every 2 s after it, against the latest quotes):

1. Recompute fair and sigmas.
2. For each symbol with a fresh quote: if `ask < fair − edge` buy; if `bid > fair + edge` sell.
3. Size: `limit × size_fraction × min(1, |gap| / (2 · edge))`, so a gap at twice the required edge gets full
   size, smaller gaps get proportionally less. `size_fraction` is 1.0 after the round-3 final (settlement is
   known), 0.3 after the round-3 prelim, 0 in rounds 1–2 by default.
4. Orders are limit orders priced 0.1% through the best quote, cancelled after 8 s if unfilled.
5. Room on each side = limit − position − resting orders on that side, so a burst can never breach a limit.

**Why rounds 1–2 are off.** Replay attribution by round/stage/symbol showed round-3 trades are profitable
in every season tested; round-1/2 trades were net negative. The measured reason: when a report arrives in
round 1 or 2, the house moves only 5% (prelim) to ~25–35% (final) of our fair-value change, because its
price already encoded the storm track. Trading our full change against it is trading noise. The code keeps a
"delta mode" for those rounds (house's pre-report mid + 25% of our change, for 60 s after the report) behind
size knobs that default to zero, in case the live house proves slower than the testnet house.

### 4.2 Strategy B: ETF arbitrage

Every 2 s, for ARCH and REBLD: compare the ETF quote with the basket of stock quotes. If buying the ETF and
selling the basket (or the reverse) nets more than 0.60 $ per unit after crossing every spread:

1. Send the ETF leg only, limit crossing by 40% of the apparent profit.
2. When the position poll confirms the fill, call `/etf/{sym}/redeem` (or `create`) and send the stock legs.
3. If the ETF leg does not fill, nothing else is sent. No naked stock exposure from a missed leg.

Capped at 10 units per arb and 6 arbs per minute. The edge exists because the house's ETF price lags its
own stock prices (std of ETF − basket in the house consensus: 2.0 $ for ARCH, 0.7 for REBLD).

### 4.3 Strategy C: pre-report lean (off by default)

After two consecutive official "direct" advisories at advisory number ≤ 4 (precision ≥ 0.75 from the research
table): short PORT, long RECON, short HAVEN (or REIN if the season loss is already above HAVEN's layer), at
30% of limits, once per storm. In replay it averaged +216 per season with a worst case of −2.3k, because the
house reacts to advisories within seconds. Enable with `--c` only if the live house is visibly slow.

### 4.4 Plumbing (most of the code)

- **Feed.** One WebSocket, one subscribe message (news + quotes, all 12 symbols). Frames go into a queue; the
  main loop drains it and handles bursts at once. Dedupe on `seq`. Auto-reconnect with 1→15 s backoff. The
  30-second `snapshot` carries every damage report of the round, so a report missed during a disconnect is
  applied from the snapshot. Season rollover (round number drops back to 1) resets all state.
- **State.** Round, phase, cumulative final damage, season loss, exposure, pending prelim, latest advisory,
  best bid/ask per symbol with timestamp, positions, our open orders. Positions and open orders are polled
  every 3 s as the source of truth; order responses update them optimistically in between.
- **Rate limiter.** Sliding one-minute window over every order action (place, cancel, cancel-all, create,
  redeem). Budget 150/min; Strategy B refuses to start if fewer than 40 remain.
- **Orders** are sent from a thread pool so the round-3 burst (up to 12 symbols) goes out in parallel.
- **Logging.** `bot.log` has every message type count, every order, fill and cancel with timestamps, and a PnL
  line every 10 s. `bot_state.json` is rewritten every 2 s for `manual.py`.
- **Replay engine.** `--dry-run N` feeds a historical season through the same `Bot` class with a fake exchange
  that builds quotes from `market.csv` (mid ± 0.15%), applies 2 s latency, and fills crossing orders at the
  quote with unlimited depth. PnL is marked at that season's real settlement and attributed per strategy,
  round, report stage and symbol.

---

## 5. Results (replay, 80 seasons, default config)

| | mean / season | median | seasons positive | worst |
|---|---|---|---|---|
| Total | +2,171 | +1,817 | 79 / 80 | −312 |
| Strategy A | +908 | +506 | 80 / 80 | +5 |
| Strategy B | +1,263 | +1,189 | 77 / 80 | −769 |

Most of A's profit: HAVEN and PORT after the round-3 final, then ARCH, AQUA, ISLE, GRID, RECON.

Sensitivity: with 4 s latency instead of 2 s the mean drops to ~+1,100; with a 0.8% spread to ~+2,000.

**Caveats, in order of importance.**

1. Never run against the real exchange. The live path was exercised against a local mock written from the
   starter kit; the response formats of `/positions`, `/orders/open` and the quote frames are guesses with
   tolerant parsers.
2. Unlimited depth at the quote. The real house will not fill 200 PORT at a stale price, and other teams are
   racing for the same quote. Expect a fraction of the replay number.
3. The model assumes the live simulator uses the same formulas as the 600 research seasons. An
   *announcement* that changes a company's economics breaks that name until adjusted by hand.
4. Profit is concentrated in a ~40-second window at the end of round 3. A disconnect or a saturated rate
   limiter at that moment costs most of the season.

---

## 6. Running it

```bash
export CTC_API_KEY=your_bot_key          # from the Team page; the bot exits with a message if missing
python3 bot.py --size 0.2                # testnet: 20% of position limits
python3 bot.py                           # competition: full size
python3 manual.py                        # second terminal: manual trades, 'fair', 'pos', 'cancel all'
```

First minute on the testnet, in `bot.log`: `exchange clock`, `raw positions payload`, `raw open_orders
payload`, `websocket connected and subscribed`, then `'quote': N` growing in the 10-second PnL line. If the
quote count stays at zero or the payload shapes look unlike `[{"symbol","quantity"}]`, the parsers in
`exchange.py` need adjusting before anything else.

Optional feed check (starter kit): `python3 starter_kit/record_feed.py --url wss://cornelltradingcompetition.org --key $CTC_API_KEY --out recording.jsonl`,
then after a season `python3 starter_kit/check_recording.py recording.jsonl history/`.

---

## 7. Adjusting during the competition

All knobs are in the `CONFIG` block at the top of `bot.py`. Restart the bot after a change (Ctrl-C, then
start again; it re-polls positions and rebuilds the round from the next snapshot within 30 s).

| Situation | Knob |
|---|---|
| House is visibly slow to move after reports (tens of seconds, as the brief claims) | `SIZE_FINAL_R12 = 0.3`, `SIZE_PRELIM_R12 = 0.2`; consider `STRAT_C = True` |
| House moves within seconds (like the testnet) | leave defaults |
| Orders rejected for position limits | `cancel all` in manual.py; the 3 s poll re-syncs |
| `RATE LIMIT` lines appear | lower `ETF_ARB_MAX_PER_MIN` or `ETF_ARB_UNITS`; raise `RECHECK_SECS` |
| Too many small round-3 trades on noisy names (GEN, FERRY) | raise `EDGE_K` from 1.5 to 2.0 |
| House quotes thin size | lower `SIZE_FINAL_R3` or send repeated rechecks (they already run every 2 s) |
| ETF arb legs keep missing | raise `ETF_ARB_CROSS_SHARE` (0.4 → 0.6) or `ETF_ARB_THRESHOLD` |
| Spreads much wider than 0.3% | raise `MIN_EDGE_FRAC` |
| An announcement changes a company | do not restart; trade it by hand with `manual.py` (`fair`, then `buy`/`sell`); the model's fair for that name is no longer trustworthy |
| Testnet rolls to a new season | automatic (round resets to 1); check `NEW SEASON detected` in the log |

Manual overrides: `buy PORT 50 71.2`, `sell HAVEN 100` (no price = cross the spread), `flat GRID`,
`flat all`, `cancel all`, `fair` (fair values, sigma, quotes, positions), `create ARCH 2`, `redeem REBLD 1`.

Manual trades share the team's 180/min order budget; the bot leaves ~30/min free.

---

## 8. What would make it better, given more time

1. A live track-based damage estimate for the current storm (storm position, radius and heading versus site
   coordinates), so rounds 1–2 have a calibrated edge instead of being switched off.
2. Reading order-book depth and sizing round-3 orders to what the house actually shows at its quote.
3. Ordering the round-3 burst by expected profit (HAVEN, PORT, ARCH first) if request latency turns out to
   be material.
4. Measuring the live house's reaction time in the first round and flipping the round-1/2 knobs automatically.
