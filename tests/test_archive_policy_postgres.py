"""Policy and concurrent-write safeguards against the dedicated CI database."""

from contextlib import closing
from datetime import timedelta
import shutil

import psycopg2
import pytest

from test_archive_postgres import database as database, insert
from research import archive
from scripts import prune_archive


def create_aggregate(conn, width="1 minute"):
    with conn.cursor() as cur:
        cur.execute(
            "CREATE MATERIALIZED VIEW klines_policy WITH (timescaledb.continuous) AS "
            "SELECT symbol, time_bucket(%s::interval, open_time) AS bucket, "
            "sum(volume) AS volume FROM klines GROUP BY symbol, bucket WITH NO DATA",
            (width,),
        )
    conn.commit()


def test_native_retention_policy_blocks_pruning(database):
    conn, _ = database
    with conn.cursor() as cur:
        # Even if a scheduler runs this policy, no fixture rows can be removed.
        cur.execute("SELECT add_retention_policy('klines', INTERVAL '200 years')")
    with pytest.raises(ValueError, match="native retention"):
        prune_archive.check_policies(conn, 7)


@pytest.mark.parametrize("start_offset,allowed", [("3 days", True), ("7 days", False)])
def test_refresh_policy_must_stay_within_preserved_raw_window(database, start_offset, allowed):
    conn, _ = database
    create_aggregate(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT add_continuous_aggregate_policy('klines_policy', "
            "start_offset => %s::interval, end_offset => INTERVAL '1 minute', "
            "schedule_interval => INTERVAL '1 hour')",
            (start_offset,),
        )
    if allowed:
        policies = prune_archive.check_policies(conn, 7)
        assert any(name == "policy_refresh_continuous_aggregate" for _, name, _, _ in policies)
    else:
        with pytest.raises(ValueError, match="reaches the pruning window"):
            prune_archive.check_policies(conn, 7)


@pytest.mark.parametrize("width,start,end,reason", [
    ("1 month", "2020-01-01", "2020-02-01", "calendar"),
    # Thursday bounds align to the Unix epoch, but not time_bucket's Monday origin.
    ("1 week", "2020-01-02", "2020-01-09", "do not align"),
])
def test_calendar_and_misaligned_week_buckets_require_manual_review(
    database, width, start, end, reason
):
    conn, _ = database
    create_aggregate(conn, width)
    with pytest.raises(ValueError, match=reason):
        prune_archive.aggregate_plan(conn, [(archive.utc(start), archive.utc(end))])


def test_raw_write_is_blocked_while_other_connection_refreshes_aggregate(
    database, tmp_path, monkeypatch
):
    conn, dsn = database
    root, backup = tmp_path / "archive", tmp_path / "backup"
    start = archive.utc("2020-01-01")
    end = start + timedelta(days=1)
    insert(conn, "BTCUSDT", start, 60)
    create_aggregate(conn)
    archive.export_range(conn, "BTCUSDT", start, end, root, batch_size=17)
    conn.rollback()  # Release the export cursor's read transaction before deletion.
    shutil.copytree(root, backup)
    connections, write_blocked = [], []

    def connect(**kwargs):
        if connections:
            # The second connection is opened for CALL while the first holds SHARE.
            with closing(psycopg2.connect(dsn)) as writer:
                with writer.cursor() as cur:
                    cur.execute("SET LOCAL lock_timeout = '100ms'")
                    with pytest.raises(psycopg2.errors.LockNotAvailable):
                        cur.execute(
                            "UPDATE klines SET volume = volume + 1 WHERE open_time = %s",
                            (start,),
                        )
                writer.rollback()
                write_blocked.append(True)
        connection = psycopg2.connect(
            dsn,
            options="-c default_transaction_read_only=on" if kwargs.get("read_only") else "",
        )
        connections.append(connection)
        return connection

    monkeypatch.setattr(prune_archive, "connect", connect)
    assert prune_archive.prune(root, backup, apply=True)
    assert write_blocked == [True] and all(connection.closed for connection in connections)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM klines")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*), sum(volume) FROM klines_policy")
        assert cur.fetchone() == (1, 60 * 1.25)
