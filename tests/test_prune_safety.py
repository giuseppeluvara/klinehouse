"""Safety regressions for verified backups and genuinely read-only pruning plans."""
from datetime import timedelta
import os
import shutil

import duckdb
import pytest

from research import archive
from research.archive_backup import backup_archive, verify_copy
from scripts import prune_archive

START = archive.utc('2020-01-01')
END = START + timedelta(seconds=2)


def rows(symbol='BTCUSDT'):
    return [(symbol, START + timedelta(seconds=i), START + timedelta(seconds=i, milliseconds=999),
             100., 102., 99., 101., 1., 101., 3, .5, 50.5) for i in range(2)]


def make_archive(root, symbols=('BTCUSDT', 'ETHUSDT')):
    for symbol in symbols:
        archive.export_batches(root, symbol, START, END, [rows(symbol)])


class Database:
    def __init__(self, *, policies=(), interval_safe=True, stale=False):
        self.policies, self.interval_safe, self.stale = policies, interval_safe, stale
        self.statements, self.sessions = [], []

    def cursor(self, **kwargs):
        db = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def execute(self, sql, params=None):
                self.sql, self.params = sql, params
                db.statements.append(sql)

            def fetchall(self):
                if 'timescaledb_information.jobs' in self.sql:
                    return db.policies
                if 'range_start' in self.sql:
                    return [(START, END)]
                if 'DISTINCT symbol' in self.sql:
                    return [('BTCUSDT',), ('ETHUSDT',)]
                return []

            def fetchone(self):
                return (db.interval_safe,)

        return Cursor()

    def set_session(self, **kwargs):
        self.sessions.append(kwargs)

    def close(self):
        pass


def test_backup_is_verified_independent_snapshot_and_excludes_orphans(tmp_path):
    root, backup = tmp_path / 'archive', tmp_path / 'backup'
    make_archive(root)
    orphan = root / 'binance_spot/BTCUSDT/2020-01/orphan.parquet'
    orphan.write_bytes(b'not published')
    report = backup_archive(root, backup)
    assert report['files'] == 2 and report['rows'] == 4
    assert not (backup / orphan.relative_to(root)).exists()
    for entry in archive.read_catalog(root):
        assert not archive.file_path(root, entry).samefile(archive.file_path(backup, entry))
        verify_copy(root, backup, entry, entry)
    with pytest.raises(FileExistsError):
        backup_archive(root, backup)


def test_hard_link_is_not_a_backup(tmp_path):
    root, backup = tmp_path / 'archive', tmp_path / 'backup'
    make_archive(root)
    shutil.copytree(root, backup, copy_function=os.link)
    entry = archive.read_catalog(root)[0]
    with pytest.raises(ValueError, match='hard links'):
        verify_copy(root, backup, entry, entry)


def test_failed_backup_does_not_publish_partial_snapshot(tmp_path, monkeypatch):
    root, backup = tmp_path / 'archive', tmp_path / 'backup'
    make_archive(root)
    from research import archive_backup
    def fail(*_):
        raise ValueError('corrupt copy')
    monkeypatch.setattr(archive_backup, 'verify_copy', fail)
    with pytest.raises(ValueError, match='corrupt'):
        backup_archive(root, backup)
    assert not backup.exists()
    assert not list(tmp_path.glob('.archive-backup-*'))
    assert len(archive.read_catalog(root)) == 2


@pytest.mark.parametrize('name,config,safe,message', [
    ('policy_retention', {'drop_after': '30 days'}, True, 'retention'),
    ('policy_refresh_continuous_aggregate', {'start_offset': None, 'end_offset': '1m'}, True, 'unbounded'),
    ('policy_refresh_continuous_aggregate', {'start_offset': '8 days', 'end_offset': '1m'}, False, 'pruning window'),
    ('custom_remove', {}, True, 'unknown scheduled'),
])
def test_unsafe_policies_fail_closed(name, config, safe, message):
    db = Database(policies=[(12, name, True, config)], interval_safe=safe)
    with pytest.raises(ValueError, match=message):
        prune_archive.check_policies(db, 7)


def test_safe_refresh_and_disabled_retention_are_read_only():
    db = Database(policies=[(1, 'policy_retention', False, {}),
                            (2, 'policy_refresh_continuous_aggregate', True,
                             {'start_offset': '3 days', 'end_offset': '1 day'})])
    assert len(prune_archive.check_policies(db, 7)) == 2
    assert not any('alter_job' in sql or 'remove_' in sql for sql in db.statements)


def test_dry_run_compares_all_symbols_and_current_values(tmp_path, monkeypatch):
    root, backup = tmp_path / 'archive', tmp_path / 'backup'
    make_archive(root)
    backup_archive(root, backup)
    database = Database()
    monkeypatch.setattr(prune_archive, 'connect', lambda **kw: database)
    def batches(conn, symbol, *_):
        data = rows(symbol)
        if database.stale and symbol == 'ETHUSDT':
            altered = list(data[0])
            altered[6] += 1
            data[0] = tuple(altered)
        yield data
    monkeypatch.setattr(prune_archive, 'db_batches', batches)
    plan = prune_archive.prune(root, backup)
    assert plan[0]['symbols'] == ['BTCUSDT', 'ETHUSDT']
    assert plan[0]['rows'] == 4 and plan[0]['source_verified']
    assert database.sessions[0] == {'isolation_level': 'REPEATABLE READ', 'readonly': True}
    assert not any('drop_chunks' in sql or 'CALL ' in sql or 'LOCK TABLE' in sql for sql in database.statements)
    database.stale = True
    with pytest.raises(ValueError, match='differs from current DB'):
        prune_archive.prune(root, backup)


@pytest.mark.parametrize('unsafe_view', [None, 'klines_1m', 'klines_1h'])
def test_policy_discovery_resolves_public_views_and_materialization_dependencies(unsafe_view):
    # Execute the discovery SQL against information-schema-shaped tables. A
    # canned cursor response cannot catch a join against the wrong relation.
    with duckdb.connect() as engine:
        engine.execute('CREATE SCHEMA timescaledb_information')
        engine.execute(
            'CREATE TABLE timescaledb_information.continuous_aggregates ('
            'hypertable_schema TEXT, hypertable_name TEXT, '
            'materialization_hypertable_schema TEXT, materialization_hypertable_name TEXT, '
            'view_schema TEXT, view_name TEXT)'
        )
        engine.execute(
            "INSERT INTO timescaledb_information.continuous_aggregates VALUES "
            "('public','klines','_timescaledb_internal','_mat_1','public','klines_1m'),"
            "('_timescaledb_internal','_mat_1','_timescaledb_internal','_mat_2','public','klines_1h'),"
            "('public','unrelated','_timescaledb_internal','_mat_3','public','unrelated_1m')"
        )
        engine.execute(
            'CREATE TABLE timescaledb_information.jobs ('
            'job_id INTEGER, proc_name TEXT, scheduled BOOLEAN, '
            'config STRUCT(start_offset VARCHAR, end_offset VARCHAR), '
            'hypertable_schema TEXT, hypertable_name TEXT)'
        )
        engine.execute(
            "INSERT INTO timescaledb_information.jobs VALUES "
            "(1,'policy_retention',false,NULL,'public','klines'),"
            "(4,'policy_compression',true,NULL,'_timescaledb_internal','_mat_1'),"
            "(5,'policy_retention',true,NULL,'public','unrelated_1m')"
        )
        for job_id, view in [(2, 'klines_1m'), (3, 'klines_1h')]:
            engine.execute(
                "INSERT INTO timescaledb_information.jobs VALUES "
                "(?,'policy_refresh_continuous_aggregate',true,?,'public',?)",
                [job_id, {'start_offset': '8 days' if view == unsafe_view else '3 days',
                          'end_offset': '1 minute'}, view],
            )

        class Cursor:
            def __enter__(self):
                self.cursor = engine.cursor()
                return self

            def __exit__(self, *_):
                self.cursor.close()

            def execute(self, sql, parameters=None):
                # Only adapt DB-API placeholders and the equivalent zero literal.
                sql = sql.replace('%s', '?').replace("interval '0'", "interval '0 seconds'")
                self.cursor.execute(sql, parameters or [])

            def fetchall(self):
                return self.cursor.fetchall()

            def fetchone(self):
                return self.cursor.fetchone()

        class Connection:
            cursor = Cursor

        if unsafe_view:
            with pytest.raises(ValueError, match='reaches the pruning window'):
                prune_archive.check_policies(Connection(), 7)
        else:
            policies = prune_archive.check_policies(Connection(), 7)
            assert {row[0] for row in policies} == {1, 2, 3, 4}
