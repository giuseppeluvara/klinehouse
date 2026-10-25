# Lightweight 1-second storage

The collector still writes closed Binance Spot `1s` candles to TimescaleDB through
Redis. The new archive stores all 12 original columns, with float64 prices and UTC
microsecond timestamps, in Zstandard Parquet. It does not contain trade ticks or
order-book events: its finest original unit remains the one-second candle.

History lives in files; DuckDB queries those files directly. Only the recent tail
needs PostgreSQL. No new database server is needed for research.

## Install and archive existing history

Run these commands from the repository on the machine holding the live DB:

```bash
git pull --ff-only
python -m pip install -r requirements-storage.txt
python scripts/archive_market.py --root archive sync
python scripts/archive_market.py --root archive verify
```

The connection uses the same `.env` and host-side `POSTGRES_PORT_HOST` as
`research/db.py`, with optional `PGHOST`/`PGPORT` overrides. Start with a bounded read-only inventory:

```bash
python scripts/audit_storage.py --root archive --output research/out/storage-audit.json
```

This saves connection availability, indexed symbol/time bounds, database and chunk
sizes, compression status and installed policies without printing credentials.
An unavailable database is reported explicitly; public Binance downloads do not
constitute evidence about local migration.

 Passwords are passed as
connection parameters, so special characters are supported. Export uses read-only
connections and server-side cursors; default batches contain at most 50,000 rows.

`sync` discovers all symbols actually present in `klines`, including symbols no
longer configured for live collection. It archives closed UTC days, skips complete
catalogued ranges and retries incomplete days so a later backfill can fill gaps.
Today's data remains in PostgreSQL. To restrict the initial migration:

```bash
python scripts/archive_market.py --root archive sync \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT --start 2026-07-01 --end 2026-08-01
```

This path requires one-second source candles and refuses other candle durations.
Do not mix collection intervals in the same `klines` table.

## Publication, integrity and recovery

Files are partitioned by market, symbol and month, with immutable daily generation
filenames. Each file is streamed back and compared with the complete source-row
fingerprint before publication. `catalog.json` records file and logical SHA-256,
row counts, bounds, gap counts, completeness and size. It is replaced atomically
after the file has been flushed and synced to disk.

A crash may leave an unreferenced generation; it never makes a partially written
file visible to readers. `gc` removes only unreferenced archive generations:

```bash
python scripts/archive_market.py --root archive gc
```

Files missing seconds remain incomplete, without invented candles. A backfill
replacement must preserve every previously archived timestamp, including when the
new source has equal or greater cardinality. Nonfinite and null values are refused. Readers retain
the archived observations and query PostgreSQL for the uncovered/incomplete
ranges; a DB row wins on an overlapping `(symbol, open_time)` key. Complete archive
ranges do not hit PostgreSQL. Ordinary reads check file presence and size;
`verify` performs the full content/checksum audit. Treat published files as
immutable and run `verify` after moving/restoring an archive.

File locking protects catalog updates, readers and compaction on one local
filesystem. Use a local APFS/ext4-style filesystem with reliable flock/atomic
rename semantics; copying or syncing the directory while a writer runs is not a
consistent backup. Stop the archive worker during a backup, or take a filesystem
snapshot. The live collector can continue running.

## Query only what is needed

```python
from research.market import load_market, iter_market

# Only these columns are read; aggregate on disk before constructing a DataFrame.
frame = load_market(
    "BTCUSDT", "2026-07-01", "2026-08-01",
    archive_root="archive", timeframe="1m",
    columns=["open_time", "close", "volume", "taker_buy_base"],
)

# For large raw ranges, consume batches and discard each one after use.
from contextlib import closing

with closing(iter_market(
    "BTCUSDT", "2026-07-01", "2026-08-01",
    archive_root="archive", timeframe="1s",
    columns=["open_time", "close"], batch_size=50_000,
)) as batches:
    for batch in batches:
        process(batch)  # replace with your incremental calculation
```

`source="archive"` supports disconnected/offline work. `source="database"` bypasses
the archive. The default `auto` merges the two. The market is Binance Spot, the
source interval is `1s`, and ranges are inclusive start/exclusive end in UTC.

Supported outputs: `1s`, `1m`, `5m`, `15m`, `1h`, `4h`, `1d`. Aggregation preserves
OHLCV, trade counts and both taker-buy volume fields. Only complete, UTC-aligned
buckets are returned: a missing second or unfinished final minute is never silently
accepted. All aggregation inputs precede the exclusive query end.

`load_market` refuses results over 1,000,000 rows by default. Use `iter_market` or
`iter_market_windows` for larger jobs. DuckDB uses a 256 MB buffer budget and two
threads by default, with spill to a temporary directory. Python/Arrow allocations
and returned DataFrames are additional memory, not covered by the DuckDB limit.
Set `MARKET_TEMP_DIR` to a directory on an SSD with free space; each query permits
up to 20 GB of spill. The recent DB tail is streamed into a temporary Parquet file,
removed on completion, so a query cannot balloon RAM by fetching all live rows.

Set `ARCHIVE_ROOT=archive` in `.env` to route the existing `load_klines` convenience
function through this archive. In that mode it requires explicit start/end.
Without it, the legacy DB-only reader still works, with a 1,000,000-row guard and
optional column projection. Existing research commands which intentionally need
more raw data should migrate to streaming or aggregate first; do not simply raise
the guard without checking memory.

The frozen sequence experiment, its protocol, data, models and output artifacts
are unchanged. This change does not introduce a Timestral trading strategy.

## Run archival automatically

For a host process, use a service manager to run:

```bash
python scripts/archive_market.py --root archive sync --loop --poll-seconds 3600
```

Or enable the optional Compose worker. Create the archive directory and set
`ARCHIVE_USER` in `.env` to your numeric user/group (`id -u` and `id -g`) so the
container can write the host directory:

```bash
mkdir -p archive
docker compose --profile archive up -d --build archive
docker compose logs -f archive
```

`ARCHIVE_ROOT` is the host bind-mount location; the worker sees it as `/archive`.
The archive dependencies are separate from the collector image. Worker failure
leaves the source data intact. No worker command applies retention or pruning.

## Compact closed months

```bash
python scripts/archive_market.py --root archive compact --symbol BTCUSDT --month 2026-07
```

A full, gap-free closed month is required. Compaction streams the existing files,
verifies the merged file and atomically switches the catalog before removing old
files. Active readers hold shared locks, so their files cannot disappear mid-query.
Repeat for other symbols. Re-exporting a daily slice inside a compacted month is
refused; the regular sync skips its complete coverage. Back up the new generation
before subsequent pruning.

## Bound the live database to approximately seven days

Keep `RETENTION_PERIOD` empty: the old native retention job does not know whether
a chunk has been archived. An existing Timescale retention job also stays active
when an environment variable is cleared; inspect/remove that policy before this
migration if one was previously installed. Do not turn it on as a shortcut.

First complete sync and verification. Create a **new** independent backup snapshot:

```bash
python scripts/backup_archive.py --root archive --destination /path/to/archive-backup
python scripts/prune_archive.py --root archive --backup-root /path/to/archive-backup
```

The copy command holds the source archive lock, copies only catalogued files,
verifies every file, syncs the copy and publishes it by directory rename. It never
overwrites an existing destination. A separate device is preferable; a same-device
copy is explicitly identified and does not protect against device failure.
Hard-linked files and overlapping source/backup paths are refused.

The default prune command is now a **verified dry-run**, not an age-only listing:
it checks archive and backup content/checksums and compares every stored column
against the current database for every symbol in each eligible chunk. It reads a
repeatable-read snapshot without refresh, table-write locks or deletion. Its JSON
plan records symbols, rows, source/backup verification and policies checked. A dry
run does not authorize later deletion and can become stale as backfills arrive.

Both modes refuse scheduled native retention on the raw table or its dependent
aggregates, unknown relevant jobs, and unbounded or too-wide refresh windows.
Fixed UTC aggregate buckets must align with candidate chunk boundaries;
calendar, timezone/custom-origin and hierarchical aggregates require manual review.
The tool only reports these blocks; it does not alter policies. Clearing `.env`
never removes an already installed Timescale policy.

The opt-in apply path remains separate. It revalidates each candidate with a
write-blocking raw-table lock, refreshes aggregates through a second connection
while keeping that lock, and drops only the verified whole chunk. A refresh/lock
failure aborts that chunk. Integration tests exercise this only on a disposable
`*_test` database. **No production apply is part of the October 2026 continuation.**
Earlier successfully committed chunks remain removed if a later chunk fails.
Policy administration must not race pruning; future manual refresh of already
pruned history can still erase materializations and must be avoided.

Keep at least seven days, beyond the built-in refresh windows (up to three days).
Historical raw data and new complete aggregates remain available through the
Parquet reader. Native backfill tools see only the hot DB; avoid reseeding pruned
periods into it. The SQL `realized_vol_1m` view is a live raw-data view, not a
historical materialization: use archived second candles for old volatility.

Disk savings follow whole chunk boundaries, not an exact rolling seven-day row
count. This code does not activate pruning automatically or change existing DB
volumes, native policies, Grafana queries, collector queues or frozen experiments.

## Validation

```bash
python -m pytest -q tests/test_archive*.py tests/test_prune_safety.py
```

The PostgreSQL tests require `TEST_POSTGRES_DSN` pointing to a dedicated database
whose name ends in `_test`. They create/drop its `klines` hypertable and test real
server-side export, archive/live merging, refusal of stale source data, verified
chunk removal, and preservation of continuous aggregates. CI supplies an isolated
TimescaleDB service. The pure storage tests need no database and cover lossless
round trips, bounded reads, gaps, flow aggregates, interrupted publication,
compaction, corruption, duplicate keys and pruning safety checks.
