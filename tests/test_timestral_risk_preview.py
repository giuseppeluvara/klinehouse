"""The single-origin preview does not read future outcomes or reuse stale fits."""

from datetime import timedelta
import json
import pytest
from scripts.preview_variance_forecast import past_context, preview
from research.timestral_experiment import utc, stamp


def test_reader_stops_at_origin_before_future_payload(tmp_path):
    start = utc("2026-01-01")
    path = tmp_path / "hours.jsonl"
    rows = [
        {"time": stamp(start + timedelta(hours=i)), "close": 10 + i} for i in range(730)
    ]
    # Anything beyond the first future timestamp must remain unparsed.
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\ninvalid future payload")
    context = past_context(path, start + timedelta(hours=729))
    assert len(context) == 721 and context[-1] == rows[728]
    assert context[0] == rows[8]


def test_preview_refuses_stale_models_and_protected_dates(tmp_path):
    for anchor, message in [
        ("2026-07-01", "current-month"),
        ("2026-08-01", "protected"),
        ("2026-07-01T01:00:00Z", "four-hour"),
    ]:
        with pytest.raises(ValueError, match=message):
            preview(tmp_path / "absent", {}, {}, "BTCUSDT", anchor)
