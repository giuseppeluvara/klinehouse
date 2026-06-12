#!/usr/bin/env bash
# =============================================================================
# klinehouse — restore a logical backup into a SIDE database (non-destructive).
#
# Restores a dump produced by scripts/backup.sh into a *new* database
# (default: <POSTGRES_DB>_restored) so your live data is never overwritten.
# Follows the official TimescaleDB pre/post-restore procedure.
#
# Usage:
#   scripts/restore.sh backups/klines_YYYYmmddTHHMMSSZ.dump [target_db]
#
# After verifying the restored DB, you can promote it (stop the collector, then
# rename databases) — see db/ documentation / README "Backups" section.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

DUMP="${1:?usage: scripts/restore.sh <dump-file> [target_db]}"
[ -s "$DUMP" ] || { echo "ERROR: dump '$DUMP' not found or empty" >&2; exit 1; }

get_env() { grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"'"; }
DB_USER="$(get_env POSTGRES_USER)"
DB_NAME="$(get_env POSTGRES_DB)"
SERVICE="timescaledb"
TARGET_DB="${2:-${DB_NAME}_restored}"

: "${DB_USER:?POSTGRES_USER not found in .env}"

echo "[restore] target database: $TARGET_DB (live '$DB_NAME' is left untouched)"
psql() { docker compose exec -T "$SERVICE" psql -U "$DB_USER" "$@"; }

psql -d postgres -c "DROP DATABASE IF EXISTS \"$TARGET_DB\";"
psql -d postgres -c "CREATE DATABASE \"$TARGET_DB\";"
psql -d "$TARGET_DB" -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
psql -d "$TARGET_DB" -c "SELECT timescaledb_pre_restore();"

echo "[restore] restoring data…"
docker compose exec -T "$SERVICE" pg_restore -U "$DB_USER" -d "$TARGET_DB" --no-owner < "$DUMP"

psql -d "$TARGET_DB" -c "SELECT timescaledb_post_restore();"

echo "[restore] done. Verify with:"
echo "  docker compose exec timescaledb psql -U $DB_USER -d $TARGET_DB -c \"SELECT symbol, count(*), max(open_time) FROM klines GROUP BY symbol;\""
