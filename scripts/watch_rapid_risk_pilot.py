"""Send local macOS alerts for the separate 24-hour Timestral paper pilot."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import subprocess
import time

from research.risk_pilot import DAY
from scripts.run_rapid_risk_pilot import MARK_WINDOW
from scripts.watch_risk_pilot import (
    COMMIT_GRACE_MS, POLL_SECONDS, RECOVERY_GRACE_SECONDS, ROME, notify, save_state, writer_status,
)


def entry_time(root: Path) -> int | None:
    journal = root / 'journal.sqlite'
    try:
        journal.stat()
    except FileNotFoundError:
        return None
    with closing(sqlite3.connect(journal.as_uri() + '?mode=ro', uri=True, timeout=2)) as db:
        entry = db.execute("SELECT recorded_ms FROM events WHERE event_key = 'entry' AND kind = 'rapid_entry'").fetchone()
    return entry[0] if entry else None


def terminal_notice(result: dict, threshold: float) -> tuple[str, str]:
    status = result['status']
    if status == 'missing_entry':
        return status, 'Ingresso paper Timestral a 24 ore saltato. Non è stata aperta una posizione.'
    if status == 'missing_exit':
        return status, 'Esito paper Timestral a 24 ore indisponibile. Manca la quotazione di uscita.'
    if status != 'valued':
        raise ValueError(f'stato del risultato sconosciuto: {status!r}')

    def number(value: float, *, signed: bool = False) -> str:
        return format(value, '+.2f' if signed else '.2f').replace('.', ',')

    timestral = result['arms']['timestral']
    ewma = result['arms']['ewma']
    drawdown = timestral['observed_max_drawdown_pct']
    comparison = 'sotto' if drawdown < threshold else 'ha raggiunto' if drawdown == threshold else 'sopra'
    when = datetime.fromtimestamp(result['quote_ms'] / 1000, ROME).strftime('%d/%m %H:%M')
    coverage = f"{result['marks_observed']}/{result['marks_expected']} rilevazioni"
    halted = ' Arresto per rischio attivo.' if timestral['risk_halted'] else ''
    late = ' Quotazioni di uscita arrivate oltre la finestra.' if result.get('late_exit') else ''
    both_cash = all(not any(arm.get('entry_units', [1])) for arm in (timestral, ewma))
    cash = ' Entrambi i portafogli sono rimasti liquidi.' if both_cash else ''
    return status, (
        f'{when} · PnL netto paper 24 ore. Timestral {number(timestral["net_pnl_usdt"], signed=True)} USDT '
        f'({number(timestral["net_return_pct"], signed=True)}%), '
        f'EWMA {number(ewma["net_pnl_usdt"], signed=True)} USDT '
        f'({number(ewma["net_return_pct"], signed=True)}%). '
        f'Divario {number(result["timestral_minus_ewma_pp"], signed=True)} pp. '
        f'Drawdown Timestral osservato {number(drawdown)}%, {comparison} la soglia del {threshold:g}% '
        f'({coverage}).{halted}{late}{cash}'
    )


def watch(root: Path, job: str) -> None:
    config = json.loads((root / 'protocol.json').read_text())
    state_path = root / 'mac-notification-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get('terminal_notice'):
        return

    def submit(key: str, value: str, message: str) -> bool:
        try:
            notify(message)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f'Notification submission failed: {exc}', flush=True)
            return False
        state[key] = value
        save_state(state_path, state)
        return True

    unhealthy_since = None
    previous_pid = None
    while True:
        result_path = root / 'result.json'
        try:
            result = json.loads(result_path.read_text())
        except FileNotFoundError:
            result = None
        except (OSError, ValueError) as exc:
            print(f'Result read failed: {exc}', flush=True)
            result = None
        if result is not None:
            try:
                status, message = terminal_notice(result, config.get('maximum_observed_drawdown', 0.25) * 100)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                print(f'Result validation failed: {exc}', flush=True)
            else:
                if submit('terminal_notice', status, message):
                    return
                time.sleep(POLL_SECONDS)
                continue

        try:
            pid, failure = writer_status(job, root / 'runner.lock')
        except (OSError, subprocess.SubprocessError) as exc:
            pid, failure = None, f'controllo del processo paper fallito ({exc})'
        observed = time.monotonic()
        if pid is not None and pid == previous_pid:
            unhealthy_since = None
            if state.pop('recovery_alerted', None):
                save_state(state_path, state)
        elif unhealthy_since is None:
            unhealthy_since = observed
        elif observed - unhealthy_since >= RECOVERY_GRACE_SECONDS and not state.get('recovery_alerted'):
            reason = failure or 'il processo paper continua a riavviarsi'
            submit('recovery_alerted', reason, f'Recupero Timestral a 24 ore fallito. {reason}.')
        previous_pid = pid

        now_ms = time.time_ns() // 1_000_000
        if result is None and now_ms > config['entry_deadline_ms'] + COMMIT_GRACE_MS:
            try:
                recorded = entry_time(root)
            except (OSError, sqlite3.Error) as exc:
                print(f'Journal read failed: {exc}', flush=True)
            else:
                if not recorded and submit('terminal_notice', 'missing_entry',
                                           'Ingresso paper Timestral a 24 ore saltato. '
                                           'La finestra è scaduta e il registro non contiene un ingresso.'):
                    return
                if recorded and now_ms > recorded + DAY + MARK_WINDOW and not state.get('exit_pending'):
                    submit('exit_pending', 'notified', 'Esito Timestral a 24 ore in ritardo. '
                           'Manca una valutazione di uscita verificata.')
        time.sleep(POLL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error('--root must be an absolute path')
    watch(args.root.resolve(), args.job)


if __name__ == '__main__':
    main()
