# Separate prospective depth-information research: capture and warmup active

Status recorded on **2026-10-30**, after predictor activation at **16:02 UTC**.

The preceding direction diagnosis found a short live regime in which the frozen
hourly-center reversal signal ranked directions poorly. A causal orientation
replay did not produce stable improvement over the nine historical cohorts; see
[the complete diagnosis](../direction-orientation-20261030-v3/README.md).

This next hypothesis adds a genuinely different input: visible resting bid/ask
depth. The [procedure](protocol-plan.json) was registered before the new capture
outcomes. It is separate from the running seven-day shadow pilot. **No new fitted
model or candidate forecast has yet been produced.**

## Acquisition actually executed

A bounded, anonymous GET-only collector is running locally from an immutable
runtime copy in `research/out/depth-research-20261030/`. It reads neither the
original pilot nor historical archives, models, environment files or credentials.
It collects three depth20 snapshots every five minutes, beginning ten seconds
before the anchor, plus a server-time clock probe. Requests and full-body client
receipts must precede the anchor; post-COMMIT observations establish persistence.
The source, quantity-derived features and sequence/hash evidence remain append-only.

The first trio at **2026-10-30 15:35 UTC (16:35 Europe/Rome)** passed independent
verification. All body receipts, primary commits, acknowledgement commits and the
separately synced proof precede the anchor. The minimum acknowledgement lead was
**9.098548 seconds**; the proof-write lead was **9.097863 seconds**. The independently
recalculated UTC clock-offset bound was **0.396738 seconds**.

| Asset | Best-level QI1 | Twenty-level QI20 | Spread, basis points |
| --- | ---: | ---: | ---: |
| BTCUSDT | -0.987111 | -0.985032 | 0.001291 |
| ETHUSDT | -0.948073 | -0.305621 | 0.044847 |
| SOLUSDT | -0.305614 | +0.281242 | 1.058481 |

These are captured feature values, not forecasts or accuracy results. Independent
80-digit Decimal arithmetic matched with maximum discrepancy **4.393e-51**.
The audit checked the raw JSON/body hashes, 20 valid levels per side, the config
seed, all twelve initial events and the proof's retained chain tip. Evidence:
[first capture proof](first-cycle.json), [write receipt](first-cycle-write-receipt.json),
[independent audit](first-cycle-independent-audit.json).

QI20 is `(sum_bid_quantity20-sum_ask_quantity20)/(sum_bid_quantity20+sum_ask_quantity20)`.
All features come from the **same** depth response. Asynchronous depth and
bookTicker responses are not combined. QI is static visible queue imbalance,
not trade flow, event-level order-flow imbalance or open interest. The REST
response has no event timestamp: client receipt proves availability, not the
freshness of the exchange's internal book. `lastUpdateId` is a sequence identifier.

The initial resource budget was 32 MiB with a 4 MiB reserve. A pre-fit operational
amendment below raises only the effective maximum to **64 MiB**; the reserve and
stop below 40 GiB free remain unchanged. Nothing is deleted. The end remains
**2026-11-05 11:10 UTC**. The original pilot's
service, runtime, model, protocol and journal remain unchanged. No order endpoint
or account access exists. The capture is a local process, not an installed
autostart service; its status/process must be checked rather than assumed alive.
The HTTP connect/read timeouts do not constitute an absolute total-response
deadline, but a late or clock-unhealthy observation remains explicitly ineligible.

## Capacity correction before fitting

Measured SQLite growth was **24–28 KiB per five-minute anchor**, including the
three raw/decoded books, clock evidence, acknowledgements and database overhead.
The registered 1,674-anchor capture therefore projects **39–46 MiB**, exceeding
the initial 28 MiB available after reserve. The extra audit files were only about
64 KiB and did not explain the deficit. This was a resource-planning defect, not
a predictive finding. [Measurements and assumptions](capture-storage-projection-20261030.json)
are retained; projections are not guarantees against further errors or restarts.

At **2026-10-30 16:58 UTC**, before any fitting, the collector registered an
[immutable budget amendment](budget-amendment-20261030.json). The original config,
sidecar and runtime remain byte-identical. The new monofile runtime is separately
versioned and pinned to SHA256 `c0bc1e3813fc3c34fbcd1ce43e9c85213f69da54d9f2d05d9f87605e109b08e5`;
the [legacy source](frozen-source/capture_depth_research_v1.py) remains available.
Only maximum storage changes from 32 to 64 MiB. Requests, features, clock bounds,
deadlines, minimum free space, activation, model, training cutoff and end are
unchanged. The manifest and one `resource_budget_amended` event bind the exact
legacy identity, reviewed new code, source prefix and clock evidence. An arbitrary
override or a new first activation after the cutoff is refused.

The old capture alone was interrupted during its waiting phase. The observed
interval from stop request to new registration was **10.53 seconds**, entirely
between scheduled collections. The original pilot service was never restarted;
the predictor stayed in warmup. The first amended trio at **17:00 UTC** brought
coverage to **54/54 expected books**, with zero missed anchor/asset pairs.
[Preservation and switch validation](budget-amendment-validation-20261030.json)
and the [independent run audit](post-budget-independent-run-audit-20261030T1700Z.json)
retain the evidence and effective budget separately from the legacy configured
budget. The [first amended trio audit](post-budget-first-trio-audit-20261030T1700Z.json)
validates client receipts and recorded primary commits before the anchor. Its
independent read occurred at 17:01:19 UTC, after the deadline; it does not itself
attest acknowledgement fsync before that deadline. No candidate forecast or new
coefficient exists in this source-only check.

## Registered future model and active warmup

The first scheduled anchor fixes the training cutoff at **2026-10-31 15:35 UTC
(16:35 Europe/Rome)**. Only the newly captured first 24 hours, eligible original
baseline probabilities and labels actually matured **and received** by that cutoff
may fit one shared coefficient. The model is

`q_new = sigmoid(logit(q_original_baseline) + beta*QI20)`.

The penalty is fixed at `0.25*beta²/2`, beta is bounded to [-8,8], and the issued
flat probability stays unchanged. Flat labels do not train direction; moving
observations have equal total weight per anchor across assets. At least 300 moving
labels pooled and 100 per asset are required; otherwise beta is frozen to zero
without extending the window. The coefficient must be durably frozen once before
any future candidate emission. There is no post-evaluation refitting or threshold
search.

The pure model helper, durable model-freeze, separate prospective forecast runner
and independent audit are implemented. The runner is active in fixed warmup at
`research/out/depth-information-20261030/`, from a frozen runtime copy. It has
**no training snapshot, fitted coefficient, candidate forecast or efficacy result**.
Its first-day cutoff remains unchanged. The runner performs no HTTP requests;
it reads only the two authorized prospective journals, using read-only coherent
transactions and incremental hash-chain verification. The collector and original
pilot remain separate processes. The pilot's configuration and runtime are
unchanged; the collector's explicit resource-only amendment is documented above.

Before the one allowed fit, an immutable training-data file fixes every array and
source reference. Private synced temporary files are published atomically without
replacing existing artifacts. Model recovery synchronizes the existing bytes and
directory before recording a new receipt; it never acknowledges an old forecast
retroactively. Each future forecast has its own acknowledgement and a seal that
records the observation after acknowledgement fsync. Forecast and acknowledgement
must be durable by anchor+30. The seal does not independently attest when its own
fsync completed; this limitation is explicit in the independent audit.

An unhealthy UTC/monotonic clock stops progression before anchors can become
permanently missing or the process can finish. Fitting and emission require a
verified clock probe no older than 600 seconds. Only final outcome collection and
shutdown after the last emission deadline allow 1,800 seconds, with the same clock
continuity and two-second offset bound. This avoids requiring a new probe after
capture has ended. The runner has a separate 128 MiB budget with a 4 MiB reserve
and stops below 40 GiB free, without deleting evidence.

The independent verifier imports neither runner, model core nor production
scoring. It reconstructs source prices, QI20, complete first-day membership,
the scalar ridge optimum, candidate vectors, timing and scores. A real warmup
audit verifies the source chains and absence of premature model/forecast artifacts.
At **16:09 UTC**, it verified 60 depth events, 10,833 original-pilot events,
21 accepted books and 18 original outcomes after capture activation. The candidate
still had zero models, forecasts and outcomes. Evidence: [predictor config](pipeline-config.json),
[launch observation](pipeline-launch-observation.json),
[independent warmup audit](pipeline-warmup-audit.json), and
[preservation and local validation](pipeline-validation.json).
Acquisition and engineering verification are not model-validation success.
The original baseline, matched price control and OI/position control must be
evaluated on identical future candidate emissions; missing/fallback/late cases
stay visible in coverage. Any failed asset remains a failed criterion. Neither
the warmup fit nor the already inspected orientation replay is confirmation.

The complete suite passed **280 tests**, including **54 capture tests**, nine
one-coefficient depth-model tests and **20 separate-runner tests**. These include
crash recovery, delayed fsync, forward clock
jumps, fixed-cutoff membership, unavailable original controls and exact fallback.
The independent auditor has **36 passing tests**, using its own journal fixtures,
known scalar optimum, amendment identity checks and tampering cases. The
registered-criteria report has **26 tests** for sufficiency, paired denominators,
flat outcomes and joint gains. Ruff and diff-whitespace checks also passed.
CI explicitly runs these tests.
An independent review checked the corrected crash-publication and clock-progress
guards before activation. Existing
environments were reused; no historical book dataset or AI model was downloaded.

## Applying the registered comparison

The report consumes one fresh successful independent audit. The auditor exports
only the exact scored cases, including symbol, anchor, label, target end, actual
source receipt and saved forecast hash. No outcome file is reopened afterward;
new appends cannot silently change the scored population between verification
and reporting. All four models must retain the same total and moving denominators.

The already registered minimum is **288 moving outcomes per asset**, covering
at least **two UTC dates of moving target ends**. Direction excludes flats;
three-class sum Brier and natural log-loss include them. The depth candidate must
strictly improve correct direction counts, Brier and log-loss against **both**
baseline and matched price control on **every** asset. Position-control results
remain descriptive and cannot veto that registered joint criterion.

The machine-readable states are `insufficient_evaluation`, `criteria_not_met`
and `preliminary_criteria_met`. Missing or nonfinite required metrics cannot
produce success. Coverage, missing outcomes, fallback, late confirmations and
source prefixes remain fully visible; no new coverage threshold or selected
period is introduced. A preliminary pass never ends the fixed experiment or
authorizes refitting. Even after the scheduled end plus 120 seconds, the report
only says `scheduled_window_elapsed`; actual process shutdown and full goal
completion require separate verification.

These incremental point criteria can be met while absolute directional accuracy
still falls below 50%. Such a result alone would not resolve the user's
directional-prediction objective. Absolute accuracy, realized class balance and
the limits of the prospective sample must remain explicit in the interpretation.

The [first real report](registered-criteria-warmup-report.json), generated on
**2026-10-30 at 16:35 UTC**, correctly states `insufficient_evaluation`: zero
candidate forecasts/outcomes, no fitted model and zero moving observations for
all three assets. [Validation](registered-criteria-validation.json) confirms no
model/training file was created and the frozen runner, core and plan hashes match.
This is an engineering check, not a predictive result. A separate
[input-integrity audit](source-integrity-audit-20261030T162240Z.json) checks all
30 expected books at 16:22 UTC: no gaps, repeated bodies, decreasing update IDs,
late client receipts or invalid primary confirmations. It does not certify source
event age or event-by-event book continuity.

```sh
.venv-timestral/bin/python scripts/report_depth_information.py \
  --root research/out/depth-information-20261030 \
  --output research/out/depth-information-20261030/reports/new-criteria-report.json
```

Output is optional, exclusive and restricted to the run's dedicated `reports/`
directory. It cannot write model, runtime, source journals or protected stores.

## Reproduce safely into a new directory

```sh
.venv-timestral/bin/python scripts/capture_depth_research.py init \
  --root research/out/depth-fresh
.venv-timestral/bin/python scripts/capture_depth_research.py run \
  --root research/out/depth-fresh
```

Each init refuses an existing directory. Do not replace the active run to repeat
a test. Future data cannot backfill a past book. The active collector's local
session/process details are retained separately in its runtime receipts.

To start a separate predictor using already authorized prospective sources:

```sh
.venv-timestral/bin/python scripts/run_depth_information.py init \
  --root research/out/depth-information-fresh \
  --capture-root research/out/depth-research-20261030 \
  --pilot-root research/out/shadow-pilot-20261029-v2
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv-timestral/bin/python \
  research/out/depth-information-fresh/runtime/scripts/run_depth_information.py run \
  --root research/out/depth-information-fresh
.venv-timestral/bin/python scripts/verify_depth_information.py \
  --root research/out/depth-information-fresh \
  --output research/out/depth-information-fresh/new-audit.json
```

The active runner is a local process, not an autostart service. Its launch receipt,
config and warmup audit are retained with the run. The first real freeze, future
candidate cycle and predictive comparison remain to be verified after the cutoff;
none are inferred from synthetic tests or from already inspected outcomes.

## Sources and limits

[Binance Spot market API](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market)
documents public depth and bookTicker. The depth20 request weight is five;
the [server-time endpoint](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/general)
adds one per cohort. That is sixteen weight units per five-minute trio.

[Cont et al.](https://arxiv.org/abs/1011.6402) study short-interval contemporaneous
order-flow price impact in equities; [Gould and Bonart](https://arxiv.org/abs/1512.03492)
study next-tick queue-imbalance prediction for Nasdaq stocks. These motivate
testing a different information source but do **not** establish predictive power
after a one-minute delay over this five-minute crypto target. This short new
experiment cannot establish statistical superiority, profitability or universal
cross-year performance.
