"""Independent audit: direct Parquet SQL, stdlib/NumPy; no research imports."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import fcntl
import hashlib
import json
import math
import time

import duckdb
import numpy as np

ROOT = Path.cwd()  # Run from the Klinehouse checkout.
TASK = ROOT / 'research/out/timestral-long-horizon-20261025'
RUN = TASK / 'run'
HOUR = timedelta(hours=1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def close(actual, expected, label, atol=1e-12, rtol=1e-10):
    if not math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol):
        raise AssertionError(f'{label}: {actual!r} != {expected!r}')


def main():
    started = time.perf_counter()
    assert (RUN / 'results.json').is_file(), 'run is not complete'
    protocol = json.loads((RUN / 'protocol.json').read_text())
    results = json.loads((RUN / 'results.json').read_text())
    protocol_sha = sha(RUN / 'protocol.json')
    assert protocol_sha == results['protocol_sha256']
    assert protocol_sha == json.loads((RUN / 'protocol.sha256.json').read_text())['sha256']
    config = protocol['config']
    for name, digest in protocol['code_sha256'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        code_path = ROOT / relative if len(relative.parts) > 1 else ROOT / 'research' / relative
        assert sha(code_path) == digest, f'code changed: {name}'
    output_hashes = {}
    for name, digest in results['output_sha256'].items():
        assert Path(name).name == name
        assert sha(RUN / name) == digest, f'output changed: {name}'
        output_hashes[name] = digest
    archive = Path(config['archive_root']).resolve()
    catalog_sha = sha(archive / 'catalog.json')
    assert catalog_sha == protocol['source']['catalog_sha256']
    catalog = json.loads((archive / 'catalog.json').read_text())
    source_start, evaluation, end = map(utc, (config['start'], config['evaluation_start'], config['end']))
    entries = [entry for entry in catalog['files'] if entry['symbol'] == config['symbol']
               and utc(entry['start']) < end and utc(entry['end']) > source_start]
    pinned = {entry['path']: entry for entry in protocol['source_verified_entries']}
    assert {entry['path']: entry for entry in entries} == pinned
    hourly = {}
    file_count = total_rows = total_bytes = 0
    with (archive / '.archive.lock').open('rb') as lock, duckdb.connect(config={'memory_limit': '128MB', 'threads': '1'}) as db:
        fcntl.flock(lock, fcntl.LOCK_SH)
        db.execute("SET TimeZone='UTC'")
        for entry in sorted(entries, key=lambda entry: entry['start']):
            relative = Path(entry['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            path = (archive / relative).resolve()
            assert path.is_relative_to(archive)
            assert path.stat().st_size == entry['bytes']
            assert sha(path) == entry['sha256'], f'archive checksum mismatch: {path}'
            # Each file is a bounded daily scan. Distinct count plus exact aligned
            # endpoints proves all 3,600 unique seconds of each hour are present.
            rows = db.execute("""
                SELECT symbol, date_trunc('hour', open_time) AS hour,
                    count(*), count(DISTINCT open_time), min(open_time), max(open_time),
                    count(close_time), count(close),
                    bool_and(open_time = date_trunc('second', open_time)
                        AND close_time >= open_time
                        AND close_time < open_time + INTERVAL '1 second'
                        AND isfinite(close) AND close > 0),
                    arg_max(close, open_time)
                FROM read_parquet(?, hive_partitioning=false)
                GROUP BY symbol, hour ORDER BY hour
            """, [str(path)]).fetchall()
            file_rows = 0
            for symbol, hour, count, unique, first, last, closed, priced, valid, price in rows:
                assert symbol == config['symbol']
                assert source_start <= hour < end
                assert utc(entry['start']) <= hour and hour + HOUR <= utc(entry['end'])
                assert count == unique == closed == priced == 3600
                assert first == hour and last == hour + HOUR - timedelta(seconds=1)
                assert valid is True and hour not in hourly
                hourly[hour] = float(price)
                file_rows += count
            assert file_rows == entry['rows']
            total_rows += file_rows
            total_bytes += path.stat().st_size
            file_count += 1
        assert sha(archive / 'catalog.json') == catalog_sha
        fcntl.flock(lock, fcntl.LOCK_UN)
    expected_hours = int((end - source_start).total_seconds() / 3600)
    assert sorted(hourly) == [source_start + i * HOUR for i in range(expected_hours)]
    assert total_rows == expected_hours * 3600 == results['data']['raw_rows']
    assert results['data']['valid_hours'] == expected_hours
    assert results['data']['invalid_hours'] == results['data']['entirely_missing_hours'] == 0
    predictions, outcomes = records(RUN / 'predictions.jsonl'), records(RUN / 'outcomes.jsonl')
    pred_by_id = {item['sample_id']: item for item in predictions}
    out_by_id = {item['sample_id']: item for item in outcomes}
    assert len(pred_by_id) == len(predictions) and len(out_by_id) == len(outcomes)
    assert pred_by_id.keys() == out_by_id.keys()
    assert all(item['status'] == 'scored' for item in outcomes)
    models = tuple(protocol['models'])
    horizon_report, max_price_error, max_return_error = {}, 0.0, 0.0
    for horizon in config['horizons']:
        history = records(RUN / f'calibration-{horizon}h.jsonl')
        earliest = source_start + config['lookback'] * HOUR
        while int(earliest.timestamp()) % (horizon * 3600):
            earliest += HOUR
        history_anchors = []
        anchor = earliest
        while anchor + horizon * HOUR <= end:
            history_anchors.append(anchor)
            anchor += horizon * HOUR
        assert [utc(item['anchor']) for item in history] == history_anchors
        for item in history:
            anchor, label_end = utc(item['anchor']), utc(item['label_end'])
            assert item['horizon_hours'] == horizon and label_end == anchor + horizon * HOUR
            reference_log = math.log(hourly[anchor - HOUR])
            target_log = math.log(hourly[label_end - HOUR])
            expected_return = target_log - reference_log
            expected_drift = horizon * (reference_log - math.log(hourly[anchor - config['lookback'] * HOUR])) / (config['lookback'] - 1)
            close(item['return'], expected_return, 'calibration outcome')
            close(item['drift_prediction'], expected_drift, 'calibration drift')
            close(item['drift_residual'], expected_return - expected_drift, 'calibration residual')
            assert item['phase'] == ('evaluation' if anchor >= evaluation else 'warmup')
        selected = sorted([item for item in predictions if item['horizon_hours'] == horizon], key=lambda item: item['anchor'])
        expected_anchors = []
        anchor = evaluation
        while anchor + horizon * HOUR <= end:
            expected_anchors.append(anchor)
            anchor += horizon * HOUR
        assert [utc(item['anchor']) for item in selected] == expected_anchors
        actual_returns = []
        for item in selected:
            anchor, label_end = utc(item['anchor']), utc(item['label_end'])
            actual = out_by_id[item['sample_id']]
            assert set(item['predictions']) == set(models)
            assert label_end == anchor + horizon * HOUR
            assert utc(item['input_start']) == anchor - config['lookback'] * HOUR
            assert utc(item['input_end_exclusive']) == anchor
            assert all(item[key] == actual[key] for key in ('anchor', 'label_end', 'horizon_hours'))
            raw_reference, raw_target = hourly[anchor - HOUR], hourly[label_end - HOUR]
            expected_return = math.log(raw_target) - math.log(raw_reference)
            close(item['reference_price'], raw_reference, 'reference price', atol=1e-8)
            close(item['reference_log_close'], math.log(raw_reference), 'reference log close')
            close(actual['actual_price'], raw_target, 'actual price', atol=1e-8)
            close(actual['return'], expected_return, 'actual log return')
            close(actual['return_percent'], math.expm1(expected_return) * 100, 'actual percent return')
            max_price_error = max(max_price_error, abs(item['reference_price'] - raw_reference), abs(actual['actual_price'] - raw_target))
            max_return_error = max(max_return_error, abs(actual['return'] - expected_return))
            eligible = [record for record in history if utc(record['label_end']) <= anchor]
            assert item['calibration']['count'] == len(eligible) >= config['min_calibration']
            assert utc(item['calibration']['latest_label_end']) == max(utc(record['label_end']) for record in eligible) <= anchor
            current_drift = horizon * (math.log(raw_reference) - math.log(hourly[anchor - config['lookback'] * HOUR])) / (config['lookback'] - 1)
            expected_baselines = {
                'unchanged': (0.0, np.quantile([record['return'] for record in eligible], [.05, .5, .95])),
                'local_drift': (current_drift, current_drift + np.quantile([record['drift_residual'] for record in eligible], [.05, .5, .95])),
            }
            for model, (point, quantiles) in expected_baselines.items():
                close(item['predictions'][model]['point'], point, f'{model} point')
                for actual_q, expected_q in zip(item['predictions'][model]['quantiles'], quantiles):
                    close(actual_q, expected_q, f'{model} causal empirical quantile')
            for prediction in item['predictions'].values():
                close(prediction['point_price'], math.exp(item['reference_log_close'] + prediction['point']), 'predicted price', atol=1e-8)
                close(prediction['point_return_percent'], math.expm1(prediction['point']) * 100, 'predicted percent')
            actual_returns.append(expected_return)
        summary = results['horizons'][str(horizon)]
        assert len(selected) == summary['scored_anchors'] == summary['emitted_predictions'] == summary['expected_anchors_if_complete']
        assert summary['skipped_missing_context_or_anchor'] == summary['skipped_insufficient_calibration'] == summary['invalidated_targets'] == 0
        assert [item['sample_id'] for item in selected] == summary['sample_ids']
        assert len(history) == summary['final_calibration_count']
        assert selected[0]['calibration']['count'] == summary['first_calibration_count']
        actual_array = np.asarray(actual_returns)
        for model in models:
            point = np.asarray([item['predictions'][model]['point'] for item in selected])
            quantiles = np.asarray([item['predictions'][model]['quantiles'] for item in selected])
            error = point - actual_array
            residual = actual_array[:, None] - quantiles
            pinball = np.maximum(np.asarray([.05, .5, .95]) * residual, np.asarray([-.95, -.5, -.05]) * residual)
            calculated = {'return_mae': np.abs(error).mean(), 'return_rmse': np.sqrt(np.square(error).mean()),
                          'direction_accuracy': (np.sign(point) == np.sign(actual_array)).mean(),
                          'pinball_mean': pinball.mean(),
                          'coverage90': ((quantiles[:, 0] <= actual_array) & (actual_array <= quantiles[:, 2])).mean(),
                          'interval_width90': (quantiles[:, 2] - quantiles[:, 0]).mean()}
            for name, value in calculated.items():
                close(summary['metrics'][model][name], value, f'{horizon}h {model} {name}')
        horizon_report[str(horizon)] = {'anchors': len(selected), 'calibration_records': len(history),
            'first_calibration_count': selected[0]['calibration']['count'], 'all_labels_matured': True,
            'first_anchor': selected[0]['anchor'], 'last_label_end': selected[-1]['label_end'],
            'all_four_models_share_samples': True, 'raw_endpoint_checks_passed': True,
            'causal_baseline_quantiles_recomputed': True, 'core_metrics_independently_recomputed': True}
    flat = records(RUN / 'forecast-rows.jsonl')
    assert len(flat) == len(predictions) * len(models)
    assert len({(item['sample_id'], item['model']) for item in flat}) == len(flat)
    for item in flat:
        parent = pred_by_id[item['sample_id']]
        assert item['model'] in models
        assert all(item[key] == parent[key] for key in ('anchor', 'label_end', 'horizon_hours', 'reference_price'))
        close(item['point_log_return'], parent['predictions'][item['model']]['point'], 'flat log-return')
    audit = {'status': 'passed', 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'Independent direct DuckDB SQL over physical daily Parquet files; no research reader or experiment imports, no model rerun',
        'audit_script_path': str(Path(__file__).resolve()), 'audit_script_sha256': sha(__file__),
        'protocol_sha256': protocol_sha,
        'results_sha256': sha(RUN / 'results.json'), 'catalog_sha256': catalog_sha,
        'source_files_sha256_checked': file_count, 'source_parquet_bytes': total_bytes,
        'raw_rows': total_rows, 'complete_unique_utc_hours': len(hourly),
        'saved_predictions': len(predictions), 'saved_outcomes': len(outcomes), 'flat_forecasts': len(flat),
        'protocol_and_code_hashes_checked': True, 'all_output_hashes_checked': output_hashes,
        'horizons': horizon_report, 'max_absolute_endpoint_price_difference_usdt': max_price_error,
        'max_absolute_log_return_difference': max_return_error, 'seconds': time.perf_counter() - started,
        'limits': ['Does not independently rerun Timestral weights.',
                   'Does not imply pretraining contamination is excluded or establish prospective predictive performance.']}
    with (TASK / 'independent-verification.json').open('x') as output:
        json.dump(audit, output, indent=2)
        output.write('\n')
    print(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main()
