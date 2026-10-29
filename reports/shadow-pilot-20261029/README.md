# Seven-day prospective shadow engineering pilot

The authorized pilot runs on BTCUSDT, ETHUSDT and SOLUSDT from **2026-10-29
11:10 UTC to 2026-11-05 11:10 UTC** (12:10 Europe/Rome on both dates).
The collector allows two final minutes for delayed outcome receipts and exits.
This is a technical commissioning period, not a completed statistical test.

The first real cycle with all three correction contexts active was verified on
**2026-10-29 at 12:31 UTC**. Forecasts were committed before the target began,
copied independently before entry, and matched unchanged to later outcomes.
The seven-day collection remains running.

## First complete real cycle

All times below are UTC (add one hour for Europe/Rome on this date):

| Asset | Forecast commit observed | Target interval | Outcome recorded |
| --- | --- | --- | --- |
| BTCUSDT | 12:25:10.700186 | 12:26:00–12:31:00 | 12:31:09.132445 |
| ETHUSDT | 12:25:10.707434 | 12:26:00–12:31:00 | 12:31:09.133052 |
| SOLUSDT | 12:25:10.713983 | 12:26:00–12:31:00 | 12:31:09.133592 |

Every commit preceded target entry by at least **49.286 seconds**. The separate
pre-target copy was captured at 12:25:14.636108 and its write confirmed at
12:25:14.639656, with zero outcomes for this anchor. Forecast sequences
386/388/390 and acknowledgement sequences 387/389/391 matched the later journal
byte-for-byte at the payload level and by hash. Outcome sequences were 427/428/429.

Both independent auditors passed on a consistent, read-only SQLite backup.
The first-cycle checker was committed as `a74e232` and passed CI before these
forecasts were emitted. It imports neither the live forecaster nor the general
auditor. A third calculation using 50-digit Decimal arithmetic checked 45 score
values; the largest difference was **1.11e-16**.

| Asset | Entry close | Exit close | Observed direction |
| --- | ---: | ---: | --- |
| BTCUSDT | 80098.00 | 80154.01 | Up |
| ETHUSDT | 2323.80 | 2325.08 | Up |
| SOLUSDT | 100.41 | 100.48 | Up |

Scores for this single interval, one observation per asset:

| Asset | Model | Three-class Brier | Natural log-loss | Direction correct |
| --- | --- | ---: | ---: | --- |
| BTCUSDT | Baseline | 0.483429480 | 0.678509403 | Yes |
| BTCUSDT | Price control | 0.482390809 | 0.677468406 | Yes |
| BTCUSDT | OI/positions | 0.482977004 | 0.678055918 | Yes |
| ETHUSDT | Baseline | 0.483460724 | 0.679798052 | Yes |
| ETHUSDT | Price control | 0.483240674 | 0.679577233 | Yes |
| ETHUSDT | OI/positions | 0.482028661 | 0.678360954 | Yes |
| SOLUSDT | Baseline | 0.487792108 | 0.726727292 | Yes |
| SOLUSDT | Price control | 0.490999906 | 0.730098447 | Yes |
| SOLUSDT | OI/positions | 0.485800205 | 0.724634056 | Yes |

Brier is the sum of squared errors over down/flat/up; log-loss uses the natural
logarithm. Moving-direction metrics condition on a non-flat realized outcome and
normalize up probability by down+up, with ties predicted up. All three models
predicted the observed upward direction. The OI/position arm has better point
scores than baseline here; the price control scores better on BTC. These three
correlated observations validate the live technical cycle and score arithmetic,
not stability, general superiority or profitability.

The checkpoint contains 429 events, 12 eligible forecasts and nine matured
outcomes across three anchors. Its 39 missing emissions are all declared startup
warmup. Three forecasts concern a still-unfinished target at capture time; there
are no unaccounted anchors, overdue unobserved outcomes, late commits or missing
commit acknowledgements.

Evidence:

- [First-cycle chronology, probabilities and metrics](first-real-cycle.json)
- [Pre-target copy](first-real-cycle-pre-target.json) and [write receipt](first-real-cycle-pre-target-receipt.json)
- [General journal and aggregate-score audit](first-real-cycle-general-audit.json)
- [50-digit arithmetic cross-check](first-real-cycle-decimal-audit.json)
- [Checkpoint hashes](first-real-cycle-checkpoint-manifest.json) and [verifier versions frozen before emission](first-real-cycle-verifiers.json)

The original checkpoint database is retained locally under
`research/out/shadow-pilot-20261029-v2/first-full-checkpoint/`. The live journal
continues appending. `first-cycle-verified.json` in the live run records that this
milestone has already been communicated; hourly supervision should avoid repeating
the notification and continue checking new anomalies and final completion.

## Implemented and verified

- Separate public GET-only collector, frozen inference runtime and append-only
  SQLite evidence journal. No order endpoint, API key, account access, database
  service, historical market reader or `.env` loading exists in the live path.
- Real receipt times, original public responses, source timestamps, request bounds,
  body hashes and revision versions. Requests cannot reach before activation.
- Three frozen July models: uniform-center baseline with mixed flat probability,
  matched price-only correction and open-interest/position correction. All use the
  same symbols, target, normalizations and stored July coefficients. No fitting
  or parameter selection occurs during the pilot.
- Forecasts every five minutes, issued after anchor+10 seconds and durably saved
  by anchor+30 seconds. Inputs are closed candles received before issuance. The
  target compares the close of the minute opened at anchor with the close of the
  minute opened at anchor+300: the return interval is anchor+60 to anchor+360.
  Exact Decimal equality is the flat class.
- Missing prices, an unverified/stale clock or a missed deadline produce explicit
  missing records. Missing OI/position context in any asset makes both correction
  models equal the baseline on all three assets. Outcomes are added separately,
  only after they mature. Late data never regenerate a previous forecast.
  Each forecast requires a separate acknowledgement observed after its SQLite
  FULL-synchronous commit returns. A late, clock-unhealthy or missing acknowledgement
  excludes that forecast from scores while preserving the evidence. A restarted
  process cannot certify an earlier unacknowledged commit. The UTC clock must agree
  with elapsed monotonic time before acquisition, issuance and outcome resolution.
- A separate verifier reads the ledger in a read-only SQLite snapshot, checks
  hashes and causality, reconstructs the exact receipt versions used, validates
  outcome prices, and independently recomputes scores. It does not rerun inference.
- The offline tests cover the real Runner with synthetic public
  responses, all three forecasts and matured outcomes, followed by the independent
  audit, slow commits, crashes before acknowledgement and wall-clock jumps.
  See `validation.json` for executed test counts. The frozen model also matched 279 saved July probability vectors with a
  maximum absolute difference of 3.26e-12. See `model-v2-verification.json`.

The first candidate snapshot `model.json` is a superseded preparation artifact;
**the deployed snapshot is `model-v2.json`**, SHA256
`ebe081bc39f856f6f903f6ddd30568c5d41b5408235239c7a1b59da93a170e74`.
`config-v2.json` and `service-start-v2.json` record the actual activation and deployment.
The live runtime is an isolated copy of the four source files whose hashes are
registered in that configuration; later checkout edits cannot change its models.

## Live evidence and operation

Private growing evidence is retained in:

```
/Users/dev-luv/Sites/klinehouse/research/out/shadow-pilot-20261029-v2/
```

The directory contains `events.sqlite`, `config.json`, `model.json` (a copy of
the deployed v2 snapshot), `status.json`, an isolated `runtime/`, and bounded
service logs. Do not copy only the SQLite main file while WAL writes are active;
use the read-only verifier or SQLite backup API. Original evidence is never pruned
by this pilot. The hash chain detects edits relative to retained external audit
heads; it is not an external timestamp authority or protection against replacing
the entire directory and all receipts.

```sh
# Current health, receipt freshness and outcome counts
.venv-timestral/bin/python scripts/run_shadow.py status \
  --run-dir research/out/shadow-pilot-20261029-v2

# Dedicated macOS service status
.venv-timestral/bin/python scripts/manage_shadow_service.py status \
  --run-dir research/out/shadow-pilot-20261029-v2

# Independent live audit; use a new output name for every retained audit
.venv-timestral/bin/python scripts/verify_shadow_run.py \
  --database research/out/shadow-pilot-20261029-v2/events.sqlite \
  --snapshot research/out/shadow-pilot-20261029-v2/model.json \
  --protocol research/out/shadow-pilot-20261029-v2/config.json \
  --output research/out/shadow-pilot-20261029-v2/audit-NEW-TIMESTAMP.json

# Stop only this trial; preserve every evidence file
.venv-timestral/bin/python scripts/manage_shadow_service.py stop \
  --run-dir research/out/shadow-pilot-20261029-v2
```

Service: `com.klinehouse.shadow.1793272200`. Its LaunchAgent starts at login and
restarts unsuccessful exits before completion. The runner holds a single-writer
lock, verifies the frozen runtime on every restart and records missed anchors.
`caffeinate -i -s` prevents idle sleep and, on AC power, system sleep while the
process lives; it does not survive power loss, logout or a disconnected network.
No global power setting is modified. The assertion ends when the service exits.

The storage budget is 512 MiB for the run directory. The process stops visibly
if it reaches that budget or free space falls below 40 GiB; it deletes nothing.
Before activation the internal SSD had 61.13 GiB free. No environment, container
or model was installed. HTTP polling is bounded to nine public sources per cycle,
normally every 30 seconds, with a dedicated pre-decision refresh. Identical
response bodies are deduplicated; first receipts and all observed revisions remain.

An hourly Codex heartbeat supervises health, independently audits the first full
live cycle, reports only significant changes and performs the final technical
review. Its ID is `collaudo-shadow-btc-eth-sol`. It must stop after completion,
preserve the evidence and never reopen historical holdouts or retune this trial.
App supervision depends on the desktop app being available; the LaunchAgent
collector itself does not depend on the chat remaining open.

## Warmup and remaining validation

Only data captured from activation forward can enter the pilot. A full spot
context requires 63 closed minutes; the position correction needs 13 five-minute
stock observations ending 15 minutes before issuance. The first baseline forecast
is therefore expected around 12:15 UTC, with complete position contexts around
12:25–12:30 UTC, subject to actual publication and receipt delays. Startup missing
records remain visible in coverage statistics. No pre-activation backfill is used
to shorten this warmup.

The first full **real** forecast/outcome cycle has now passed the checks above.
The seven-day report remains pending and must show all scheduled/missing forecasts,
eligible correction counts, fallback and unobserved outcomes, by asset.

The runner retries public-source failures, using a bounded 67-minute spot and
85-minute metrics recovery window after failures/restarts. A longer outage can
leave old outcomes unavailable; these stay explicitly pending and must not be
dropped from coverage denominators. Error-free HTTP responses are not sufficient
evidence of fresh data: supervision must also inspect latest source timestamps.

Report full three-class Brier and natural log-loss, conditional moving direction
accuracy, and paired score differences on identical emissions. Report data
availability, revisions and measured first-receipt delays separately. A historical
snapshot is deliberately old; this trial does not test a current rolling fit.
Seven days cannot establish stability across years, universal superiority or
profitability. Historical OI/position performance was not stably better, and live
REST-to-archive semantic equivalence remains unverified. The fixed 15-minute lag
is a registered feature rule, not a certified exchange publication bound.

## Source documentation

- [Binance Spot market API](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market)
- [Binance USD-M market API](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)

Current documentation marks OI/position timestamps as period ends. The position
endpoint documentation advertises a key requirement, but this machine's bounded
anonymous preflight returned HTTP 200; actual live response availability is audited
separately. No credentials are read or supplied if anonymous access later fails.

## Initial commissioning capture

The first service, `com.klinehouse.shadow.1793271300`, was stopped and disabled
before any forecast was emitted. Its 24 public responses and zero forecasts
remain in `research/out/shadow-pilot-20261029/`; see
`initial-capture-closed-audit.json`. A review found that a pre-commit timestamp
alone did not prove when an SQLite write completed. The corrected v2 service
adds post-commit acknowledgements and monotonic-clock checks. The full seven-day
clock restarts at the v2 activation stated above; the earlier capture is excluded.
