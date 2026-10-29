import copy
import json

import pytest

from research import shadow_sources as sources


START = 1800000000  # Synthetic timestamps, without any network or market reads.
END = START + 7 * 86400


def candle(stamp):
    return [
        stamp * 1000,
        "100",
        "101",
        "99",
        "100.5",
        "10",
        stamp * 1000 + 59999,
        "1000",
        20,
        "6",
        "600",
        "0",
    ]


def metric(source, stamp):
    key = "sumOpenInterest" if source == "open_interest" else "longShortRatio"
    return {"symbol": "BTCUSDT", "timestamp": stamp * 1000, key: "123.5"}


def test_request_windows_never_cross_activation_or_include_unclosed_candles():
    for source in sources.URLS:
        assert sources.request_parameters(source, "BTCUSDT", START, END, START) is None
        for recovery in (False, True):
            p = sources.request_parameters(
                source, "BTCUSDT", START, END, START + 803, recovery
            )
            assert p["startTime"] >= START * 1000
            assert p["limit"] <= 100
            if source == "spot_klines":
                assert p["interval"] == "1m"
                assert p["endTime"] == (START + 780) * 1000 - 1
            else:
                assert p["period"] == "5m"
                assert p["endTime"] == (START + 600) * 1000
    p = sources.request_parameters("spot_klines", "BTCUSDT", START, END, END + 60)
    assert p["endTime"] == END * 1000 - 1


@pytest.mark.parametrize(
    "source,symbol",
    [
        ("https://evil.invalid", "BTCUSDT"),
        ("orders", "BTCUSDT"),
        ("spot_klines", "BTCUSDT&signature=secret"),
    ],
)
def test_only_fixed_public_sources_and_symbols_are_allowed(source, symbol):
    with pytest.raises(ValueError, match="unregistered"):
        sources.request_parameters(source, symbol, START, END, START + 1000)


@pytest.mark.parametrize("source", tuple(sources.URLS))
def test_validate_content_times_and_registered_schema(source):
    params = sources.request_parameters(
        source, "BTCUSDT", START, END, START + 1000, recovery=True
    )
    stamp = START + (840 if source == "spot_klines" else 900)
    body = [candle(stamp) if source == "spot_klines" else metric(source, stamp)]
    assert sources.validate_body(source, body, params, START, END, START + 1000) == [
        stamp
    ]
    with pytest.raises(ValueError, match="duplicate"):
        sources.validate_body(source, body + body, params, START, END, START + 1000)
    with pytest.raises(ValueError):
        sources.validate_body(source, body, params, START, END, stamp - 1)
    with pytest.raises(ValueError):
        sources.validate_body(source, body, params, stamp + 1, END, START + 1000)
    wrong_window = dict(params, startTime=(stamp + 1) * 1000)
    with pytest.raises(ValueError, match="requested window"):
        sources.validate_body(source, body, wrong_window, START, END, START + 1000)


def test_unclosed_spot_and_invalid_ohlc_or_fractional_trades_are_rejected():
    params = sources.request_parameters(
        "spot_klines", "BTCUSDT", START, END, START + 1000, recovery=True
    )
    row = candle(START + 900)
    with pytest.raises(ValueError, match="unclosed"):
        sources.validate_body("spot_klines", [row], params, START, END, START + 959)
    for column, value in (
        (1, "0"),
        (2, "98"),
        (3, "102"),
        (4, "NaN"),
        (8, 1.5),
        (9, "11"),
    ):
        invalid = copy.deepcopy(row)
        invalid[column] = value
        with pytest.raises(ValueError):
            sources.validate_body(
                "spot_klines", [invalid], params, START, END, START + 1000
            )


@pytest.mark.parametrize("source", ("open_interest", "position_ratio"))
def test_metric_label_grid_symbol_and_positive_values(source):
    params = sources.request_parameters(
        source, "BTCUSDT", START, END, START + 1000, recovery=True
    )
    row = metric(source, START + 900)
    key = "sumOpenInterest" if source == "open_interest" else "longShortRatio"
    for column, value in (
        ("symbol", "ETHUSDT"),
        ("timestamp", (START + 901) * 1000),
        (key, "0"),
        (key, "nan"),
    ):
        invalid = {**row, column: value}
        with pytest.raises(ValueError):
            sources.validate_body(source, [invalid], params, START, END, START + 1000)


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status
        self.content = json.dumps(body).encode()

    def json(self):
        return self.body


def test_fetch_is_fixed_get_with_no_credential_headers_or_redirects(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "do-not-send-this")
    monkeypatch.setenv("BINANCE_API_SECRET", "do-not-send-that")
    calls = []
    body = [candle(START + 840)]

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response(body)

    times = iter((START + 1000, START + 1000.5))
    actual = sources.fetch(
        "spot_klines", "BTCUSDT", START, END, clock=lambda: next(times), get=get
    )
    url, kwargs = calls[0]
    assert url == sources.URLS["spot_klines"]
    assert kwargs["allow_redirects"] is False
    assert "headers" not in kwargs and "auth" not in kwargs
    assert "do-not-send" not in json.dumps(calls)
    assert actual["requested_at"] == START + 1000
    assert actual["received_at"] == START + 1000.5
    assert actual["body_sha256"] == sources.digest(body)
    assert actual["content_times"] == [START + 840]


def test_public_transport_disables_environment_and_netrc_auth(monkeypatch):
    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get(self, url, **kwargs):
            assert self.trust_env is False
            assert "auth" not in kwargs and "headers" not in kwargs
            assert kwargs["allow_redirects"] is False
            return Response([])

    monkeypatch.setattr(sources.requests, "Session", Session)
    assert (
        sources.public_get(
            sources.URLS["spot_klines"], allow_redirects=False
        ).status_code
        == 200
    )


def test_response_size_is_bounded_before_json_is_decoded():
    response = Response([])
    response.content = b" " * 100_001

    def decode():
        raise AssertionError("oversized body must not be decoded")

    response.json = decode
    with pytest.raises(ValueError, match="oversized"):
        sources.fetch(
            "open_interest",
            "BTCUSDT",
            START,
            END,
            clock=lambda: START + 1000,
            get=lambda *a, **k: response,
        )


@pytest.mark.parametrize("status", (301, 302, 401, 403, 429, 500))
def test_non_success_and_redirect_responses_fail_without_following(status):
    def get(url, **kwargs):
        assert kwargs["allow_redirects"] is False
        return Response([], status)

    with pytest.raises(RuntimeError, match=f"status {status}"):
        sources.fetch(
            "open_interest", "BTCUSDT", START, END, clock=lambda: START + 1000, get=get
        )
