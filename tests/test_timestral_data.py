"""Public-source acquisition tests: integrity, bounded parsing and immutable reuse."""

from datetime import timedelta
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import archive  # noqa: E402
from scripts import prepare_timestral_data as data  # noqa: E402

DAY = archive.utc("2026-07-01")
NAME = "BTCUSDT-1s-2026-07-01.zip"


def source_row(offset=0):
    stamp = archive.micros(DAY + timedelta(seconds=offset))
    return [
        str(stamp),
        "100.01",
        "100.03",
        "99.98",
        "100.02",
        "1.25",
        str(stamp + 999999),
        "125.025",
        "3",
        "0.75",
        "75.015",
        "0",
    ]


def zip_bytes(rows, member=None):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as target:
        text = "\n".join(",".join(row) for row in rows) + "\n"
        target.writestr(member or NAME.replace(".zip", ".csv"), text)
    return content.getvalue()


def write_zip(tmp_path, rows, member=None):
    target = tmp_path / NAME
    target.write_bytes(zip_bytes(rows, member))
    return target


def test_microsecond_columns_are_mapped_losslessly_and_batches_are_bounded(tmp_path):
    source = write_zip(tmp_path, [source_row(i) for i in range(7)])
    batches = list(data.iter_zip_batches(source, "BTCUSDT", DAY, batch_size=3))
    assert [batch.num_rows for batch in batches] == [3, 3, 1]
    assert all(batch.schema == archive.SCHEMA for batch in batches)
    rows = [row for batch in batches for row in archive.table_rows(batch)]
    assert rows[0] == (
        "BTCUSDT",
        DAY,
        DAY + timedelta(microseconds=999999),
        100.01,
        100.03,
        99.98,
        100.02,
        1.25,
        125.025,
        3,
        0.75,
        75.015,
    )
    assert rows[-1][1] == DAY + timedelta(seconds=6)


def test_legacy_milliseconds_are_explicitly_date_gated():
    day = archive.utc("2024-07-01")
    row = source_row()
    row[0] = str(archive.micros(day) // 1000)
    row[6] = str(archive.micros(day) // 1000 + 999)
    assert data.parse_candle(row, "BTCUSDT", day)[2] == day + timedelta(
        milliseconds=999
    )
    with pytest.raises(ValueError, match="source UTC day"):
        data.parse_candle(row, "BTCUSDT", DAY)


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row.__setitem__(6, str(int(row[0]) + 1_000_000)),
        lambda row: row.__setitem__(4, "nan"),
        lambda row: row.__setitem__(8, "1.5"),
        lambda row: row.__setitem__(11, "1"),
        lambda row: row.pop(),
    ],
)
def test_invalid_candles_fail_instead_of_silently_changing_source(change):
    row = source_row()
    change(row)
    with pytest.raises(ValueError):
        data.parse_candle(row, "BTCUSDT", DAY)


def test_gaps_remain_visible_and_duplicate_seconds_are_rejected(tmp_path):
    source = write_zip(tmp_path, [source_row(0), source_row(2)])
    entry = archive.export_batches(
        tmp_path / "archive",
        "BTCUSDT",
        DAY,
        DAY + timedelta(days=1),
        data.iter_zip_batches(source, "BTCUSDT", DAY, 1),
        batch_size=1,
    )
    assert entry["rows"] == 2 and entry["gaps"] == 1 and not entry["complete"]
    source.write_bytes(zip_bytes([source_row(0), source_row(0)]))
    with pytest.raises(ValueError, match="duplicate"):
        list(data.iter_zip_batches(source, "BTCUSDT", DAY))


def test_zip_members_and_checksum_names_cannot_redirect_inputs(tmp_path):
    source = write_zip(tmp_path, [source_row()], member="../wrong.csv")
    with pytest.raises(ValueError, match="CSV member"):
        list(data.iter_zip_batches(source, "BTCUSDT", DAY))
    digest = "a" * 64
    assert data.parse_checksum(f"{digest}  {NAME}\n", NAME) == digest
    with pytest.raises(ValueError, match="filename"):
        data.parse_checksum(f"{digest}  ../{NAME}\n", NAME)
    with pytest.raises(ValueError, match="one file"):
        data.parse_checksum(f"{digest}  {NAME}\n{digest}  another.zip\n", NAME)


class Response:
    def __init__(self, url, content, *, fail=False):
        self.url, self.content, self.fail = url, content, fail
        self.headers = {"Content-Length": str(len(content))}
        self.raw = io.BytesIO(content)

    def raise_for_status(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def iter_content(self, chunk_size):
        if self.fail:
            yield self.content[:10]
            raise OSError("interrupted transfer")
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]


class Session:
    def __init__(self, content, *, checksum=None, fail=False):
        self.content = content
        self.checksum = checksum or hashlib.sha256(content).hexdigest()
        self.calls, self.fail = [], fail

    def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        assert kwargs.get("verify", True) is not False
        payload = (
            f"{self.checksum}  {NAME}\n".encode()
            if url.endswith(".CHECKSUM")
            else self.content
        )
        return Response(url, payload, fail=self.fail and not url.endswith(".CHECKSUM"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_checksum_failure_and_interruption_never_publish_zip(tmp_path):
    content = zip_bytes([source_row()])
    item = {"url": data.source_url("BTCUSDT", DAY), "bytes": len(content)}
    with pytest.raises(ValueError, match="verification failed"):
        data.acquire_zip(Session(content, checksum="0" * 64), item, tmp_path)
    assert not (tmp_path / NAME).exists()
    assert not list(tmp_path.glob(".*zip-*"))
    other = tmp_path / "interrupted"
    other.mkdir()
    with pytest.raises(OSError, match="interrupted"):
        data.acquire_zip(Session(content, fail=True), item, other)
    assert not (other / NAME).exists()
    assert not list(other.glob(".*zip-*"))


def test_prepare_reuses_verified_snapshot_without_network_and_refuses_changes(
    tmp_path, monkeypatch
):
    session = Session(zip_bytes([source_row(0), source_row(2)]))
    monkeypatch.setattr(data.requests, "Session", lambda: session)
    output = tmp_path / "pilot"
    manifest = data.prepare(
        output, "BTCUSDT", DAY, DAY + timedelta(days=1), min_free_gib=0
    )
    assert manifest["source"] == "binance_public_data"
    assert manifest["local_database_verified"] is False
    assert manifest["rows"] == 2 and not manifest["complete"]
    assert manifest["files"][0]["archive_entry"]["gaps"] == 1
    saved = (output / "provenance.json").read_bytes()
    session.calls.clear()
    assert (
        data.prepare(output, "BTCUSDT", DAY, DAY + timedelta(days=1), min_free_gib=0)
        == manifest
    )
    assert not session.calls
    assert (output / "provenance.json").read_bytes() == saved
    with pytest.raises(ValueError, match="different immutable"):
        data.prepare(output, "BTCUSDT", DAY, DAY + timedelta(days=2), min_free_gib=0)
    (output / "downloads" / NAME).write_bytes(b"corruption")
    with pytest.raises(ValueError, match="ZIP changed"):
        data.prepare(output, "BTCUSDT", DAY, DAY + timedelta(days=1), min_free_gib=0)


def test_plan_is_read_only_and_budget_blocks_download(tmp_path, monkeypatch):
    session = Session(zip_bytes([source_row()]))
    monkeypatch.setattr(data.requests, "Session", lambda: session)
    output = tmp_path / "not-created"
    result = data.prepare(
        output, "BTCUSDT", DAY, DAY + timedelta(days=1), plan_only=True
    )
    assert result["download_bytes"] == len(session.content)
    assert not output.exists()
    assert all(method == "HEAD" for method, _ in session.calls)
    with pytest.raises(ValueError, match="size budget"):
        data.prepare(
            output, "BTCUSDT", DAY, DAY + timedelta(days=1), max_download_mib=0
        )
    assert not output.exists()
