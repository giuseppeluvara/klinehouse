"""Local macOS alerts for the frozen paper pilot. Reads the journal without writing to it."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile
import time
from zoneinfo import ZoneInfo

from research.risk_pilot import DAY, WEEK
from scripts.verify_risk_pilot import audit


POLL_SECONDS = 5
RECOVERY_GRACE_SECONDS = 60
COMMIT_GRACE_MS = 10_000
BOUNDARY_QUOTE_GRACE_MS = 120_000
ROME = ZoneInfo('Europe/Rome')
NOTIFICATION_APP = Path.home() / 'Library/Application Support/Klinehouse/Timestral Alerts.app'


def notify(message: str, *, test: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix='timestral-alert-') as directory:
        receipt = Path(directory) / 'receipt.json'
        subprocess.run(['/usr/bin/open', '-g', '-j', '-n', '-W', str(NOTIFICATION_APP), '--args',
                        'authorize' if test else 'notify', message, str(receipt)],
                       check=True, timeout=120 if test else 10)
        result = json.loads(receipt.read_text())
        if not result.get('accepted') or (test and not result.get('delivered')):
            raise ValueError(result.get('error', 'notification delivery was not verified'))
    print(json.dumps({'notification_submitted': message, **result}, ensure_ascii=False), flush=True)


def writer_status(job: str, lock: Path) -> tuple[int | None, str]:
    result = subprocess.run(['/bin/launchctl', 'print', job], capture_output=True,
                            text=True, timeout=3)
    if result.returncode:
        return None, 'il servizio macOS del processo paper non è disponibile'
    state = re.search(r'^\s*state = (.+)$', result.stdout, re.M)
    pid = re.search(r'^\s*pid = (\d+)$', result.stdout, re.M)
    if state and state[1] == 'running' and pid:
        opened = subprocess.run(['/usr/sbin/lsof', '-a', '-p', pid[1], '-Fn', '--', str(lock)],
                                capture_output=True, text=True, timeout=3)
        if opened.returncode == 0 and f'n{lock}' in opened.stdout.splitlines():
            return int(pid[1]), ''
    return None, 'il processo paper non ha ripreso il registro'


def entry_events(root: Path, first_anchor: int) -> tuple[dict, bool]:
    uri = (root / 'journal.sqlite').as_uri() + '?mode=ro'
    db = sqlite3.connect(uri, uri=True, timeout=2)
    try:
        rows = db.execute("SELECT event_key, kind, payload FROM events "
                          "WHERE event_key LIKE 'entry:%' OR event_key IN ('expired', ?, ?)",
                          (f'outcome:{first_anchor}', f'mark:{(first_anchor + 8 * DAY) // DAY}')).fetchall()
    finally:
        db.close()
    return {key: (kind, json.loads(payload)) for key, kind, payload in rows}, any(
        key == 'expired' for key, _, _ in rows)


def missed_entries(config: dict, entries: dict, now_ms: int):
    for anchor in range(config['first_anchor_ms'], config['last_exit_ms'] - DAY, WEEK):
        entry_ms = anchor + DAY
        event = entries.get(f'entry:{anchor}')
        if event and event[0] == 'missing_entry':
            yield anchor, entry_ms, str(event[1].get('reason', 'ingresso dichiarato mancante'))[:180]
        elif event is None and now_ms > entry_ms + config['entry_deadline_ms'] + COMMIT_GRACE_MS:
            yield anchor, entry_ms, 'la finestra è scaduta e il registro non contiene un ingresso'


def first_week_notice(root: Path, config: dict, entries: dict, now_ms: int,
                      previous: str | None) -> tuple[str, str] | None:
    anchor = config['first_anchor_ms']
    outcome = entries.get(f'outcome:{anchor}')
    if previous in ('valued', 'missing_first_entry') or not outcome or outcome[0] != 'outcome':
        return None

    boundary = anchor + 8 * DAY
    first_entry = entries.get(f'entry:{anchor}')
    next_entry = entries.get(f'entry:{anchor + WEEK}')
    boundary_mark = entries.get(f'mark:{boundary // DAY}')
    if (not first_entry or first_entry[0] != 'entry' or
            (next_entry and next_entry[0] == 'entry') or boundary_mark):
        week = audit(root)['independent_paper_accounting']['first_week']
        if week['status'] == 'missing_first_entry':
            return 'missing_first_entry', 'Primo esito paper registrato. L’ingresso iniziale è saltato, quindi il PnL è indisponibile.'
        if week['status'] == 'valued':
            timestral, ewma = (week['arms'][arm] for arm in ('timestral', 'ewma'))

            def signed(value: float) -> str:
                return f'{value:+.2f}'.replace('.', ',')

            drawdown = timestral['observed_max_drawdown_pct']
            threshold = config['maximum_observed_drawdown'] * 100
            comparison = 'sotto' if drawdown < threshold else 'a o sopra'
            dd = f'{drawdown:.2f}'.replace('.', ',')
            when = datetime.fromtimestamp(week['quote_ms'] / 1000, ROME).strftime('%d/%m %H:%M')
            coverage = f"{week['daily_marks']}/{week['expected_daily_marks']} rilevazioni"
            halted = ' Arresto per rischio attivo.' if timestral['risk_halted'] else ''
            message = (f'{when} · PnL netto paper: Timestral {signed(timestral["net_pnl_usdt"])} USDT '
                       f'({signed(timestral["net_return_pct"])}%), EWMA {signed(ewma["net_pnl_usdt"])} USDT '
                       f'({signed(ewma["net_return_pct"])}%). Divario {signed(week["timestral_minus_ewma_pp"])} pp. '
                       f'Drawdown osservato {dd}%, {comparison} il {threshold:g}% ({coverage}).{halted}')
            return 'valued', message

    if previous != 'awaiting_quotes' and now_ms >= boundary + BOUNDARY_QUOTE_GRACE_MS:
        return 'awaiting_quotes', 'Primo esito paper registrato. Attendo le quotazioni di chiusura per PnL e drawdown.'
    return None


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def watch(root: Path, job: str) -> None:
    config = json.loads((root / 'protocol.json').read_text())
    path = root / 'mac-notification-state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    sent_entries = set(state.get('missed_entries', []))
    unhealthy_since = None
    previous_pid = None
    retry_after: dict[str, float] = {}
    next_audit_after = 0.0

    def submit(message: str) -> bool:
        if time.monotonic() < retry_after.get(message, 0):
            return False
        try:
            notify(message)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f'Notification submission failed: {exc}', flush=True)
            retry_after[message] = time.monotonic() + 60
            return False
        retry_after.pop(message, None)
        return True

    print(json.dumps({'watching': job, 'poll_seconds': POLL_SECONDS}), flush=True)
    while True:
        now_ms = time.time_ns() // 1_000_000
        previous_state = dict(state)
        expired = False
        failure = ''
        try:
            entries, expired = entry_events(root, config['first_anchor_ms'])
        except (OSError, sqlite3.Error, ValueError) as exc:
            failure = f'lettura del registro paper fallita ({exc})'
        else:
            for anchor, entry_ms, reason in missed_entries(config, entries, now_ms):
                if anchor in sent_entries:
                    continue
                when = datetime.fromtimestamp(entry_ms / 1000, ROME).strftime('%d/%m alle %H:%M')
                if submit(f'Ingresso paper saltato il {when}. {reason}.'):
                    sent_entries.add(anchor)
                    state['missed_entries'] = sorted(sent_entries)
                    save_state(path, state)
            if f"outcome:{config['first_anchor_ms']}" in entries and time.monotonic() >= next_audit_after:
                try:
                    report = first_week_notice(root, config, entries, now_ms, state.get('first_week_notice'))
                except (AssertionError, KeyError, TypeError, OSError, sqlite3.Error, ValueError) as exc:
                    print(f'First outcome audit failed: {exc!r}', flush=True)
                    next_audit_after = time.monotonic() + 60
                    if state.get('first_week_notice') != 'audit_failed' and submit(
                            'Primo esito registrato. Verifica del PnL paper fallita, controllo in corso.'):
                        state['first_week_notice'] = 'audit_failed'
                else:
                    if report and submit(report[1]):
                        state['first_week_notice'] = report[0]
                        save_state(path, state)

        if not expired and now_ms <= config['last_exit_ms'] + 600_000:
            try:
                pid, writer_failure = writer_status(job, root / 'runner.lock')
            except (OSError, subprocess.SubprocessError) as exc:
                pid, writer_failure = None, f'controllo del processo paper fallito ({exc})'
            failure = failure or writer_failure
            observed = time.monotonic()
            if not failure and pid is not None and pid == previous_pid:
                unhealthy_since = None
                state.pop('recovery_alerted', None)
            elif unhealthy_since is None:
                unhealthy_since = observed
            elif observed - unhealthy_since >= RECOVERY_GRACE_SECONDS:
                reason = failure or 'il processo paper continua a riavviarsi'
                if not state.get('recovery_alerted') and submit(f'Recupero Timestral fallito. {reason}.'):
                    state['recovery_alerted'] = True
            previous_pid = pid if not failure else None

        if state != previous_state:
            save_state(path, state)
        if expired or now_ms > config['last_exit_ms'] + 600_000:
            return
        time.sleep(POLL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-notification', action='store_true')
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    if args.test_notification:
        notify('Notifica di prova. Il monitor locale di Timestral è attivo.', test=True)
    elif args.root:
        watch(args.root.resolve(), f'gui/{os.getuid()}/com.klinehouse.timestral.paper')
    else:
        parser.error('--root is required when monitoring')


if __name__ == '__main__':
    main()
