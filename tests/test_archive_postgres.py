"""Integration tests against the dedicated TimescaleDB CI service."""

from __future__ import annotations

from contextlib import closing
from datetime import timedelta
import os
from pathlib import Path
import shutil
import sys

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import archive, market  # noqa: E402
from scripts import prune_archive  # noqa: E402


@pytest.fixture
def database():
    dsn = os.environ.get("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set TEST_POSTGRES_DSN to a dedicated *_test Timescale database")
    conn = psycopg2.connect(dsn)
    assert conn.info.dbname.endswith("_test"), (
        "integration tests require an isolated *_test database"
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
        cur.execute("DROP TABLE IF EXISTS klines CASCADE")
        schema = (
            (Path(__file__).resolve().parents[1] / "db/init.sql")
            .read_text()
            .split("-- Convert to hypertable")[0]
        )
        cur.execute(schema)
        cur.execute(
            "SELECT create_hypertable('klines', 'open_time', chunk_time_interval => INTERVAL '1 day')"
        )
    conn.autocommit = False
    try:
        yield conn, dsn
    finally:
        conn.rollback()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("DROP TABLE klines CASCADE")
        conn.close()


def insert(conn, symbol, start, seconds):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO klines SELECT %s, t, t + INTERVAL '999 milliseconds', "
            "100, 102, 99, 101, 1.25, 130.5, 3, .75, 78.3 "
            "FROM generate_series(%s::timestamptz, %s::timestamptz - INTERVAL '1 second', "
            "INTERVAL '1 second') t",
            (symbol, start, start + timedelta(seconds=seconds)),
        )
    conn.commit()


def test_server_cursor_export_and_live_tail(database, tmp_path):
    conn, _ = database
    start = archive.utc("2020-01-01")
    insert(conn, "BTCUSDT", start, 180)
    entry = archive.export_range(
        conn, "BTCUSDT", start, start + timedelta(seconds=120), tmp_path, 17
    )[0]
    archive.verify_entry(tmp_path, entry)
    result = market.load_market(
        "BTCUSDT",
        start,
        start + timedelta(seconds=180),
        archive_root=tmp_path,
        conn=conn,
        timeframe="1m",
    )
    assert len(result) == 3 and result.volume.sum() == 180 * 1.25
    assert result.taker_buy_base.sum() == 180 * 0.75


def test_verified_prune_preserves_aggregates_and_refuses_stale_source(
    database, tmp_path, monkeypatch
):
    conn, dsn = database
    root, backup = tmp_path / "archive", tmp_path / "backup"
    start = archive.utc("2020-01-01")
    end = start + timedelta(days=1)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        insert(conn, symbol, start, 86400)
        archive.export_range(conn, symbol, start, end, root)
    with conn.cursor() as cur:
        cur.execute(
            "CREATE MATERIALIZED VIEW klines_1m WITH (timescaledb.continuous) AS "
            "SELECT symbol, time_bucket('1 minute', open_time) AS bucket, sum(volume) AS volume "
            "FROM klines GROUP BY symbol, bucket WITH NO DATA"
        )
    conn.commit()
    shutil.copytree(root, backup)
    monkeypatch.setattr(
        prune_archive,
        "connect",
        lambda **kwargs: psycopg2.connect(
            dsn,
            options="-c default_transaction_read_only=on"
            if kwargs.get("read_only")
            else "",
        ),
    )
    plan = prune_archive.prune(root, backup)
    assert plan and all(e["action"] == "would verify and drop" for e in plan)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE klines SET close = close + 1 WHERE symbol = 'ETHUSDT' AND open_time = %s",
            (start,),
        )
    conn.commit()
    with pytest.raises(ValueError, match="differs from current DB"):
        prune_archive.prune(root, backup, apply=True)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM klines")
        assert cur.fetchone()[0] == 172800
        cur.execute(
            "UPDATE klines SET close = close - 1 WHERE symbol = 'ETHUSDT' AND open_time = %s",
            (start,),
        )
    conn.commit()
    removed = prune_archive.prune(root, backup, apply=True)
    assert removed
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM klines")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*), sum(volume) FROM klines_1m")
        assert cur.fetchone() == (2880, 172800 * 1.25)
    # Fully archived ranges must also work with no available database.
    with closing(psycopg2.connect(dsn)) as tail:
        frame = market.load_market(
            "BTCUSDT", start, end, archive_root=root, conn=tail, timeframe="1h"
        )
        assert len(frame) == 24
