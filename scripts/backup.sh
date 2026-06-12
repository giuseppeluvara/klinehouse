#!/usr/bin/env bash
# =============================================================================
# klinehouse — logical backup of the TimescaleDB database.
#
# Dumps the database (custom format, compressed) from the running container to
# ./backups/, then prunes old backups beyond the retention count. The named
# Docker volume already protects against container/process crashes; this script
# protects against the things the volume does NOT: `docker compose down -v`,
# accidental volume removal, and disk failure.
#
# Usage:
#   scripts/backup.sh                 # one backup now (cron-friendly)
#
# Schedule with cron, e.g. daily at 03:00:
#   0 3 * * * cd /path/to/klinehouse && scripts/backup.sh >> backups/backup.log 2>&1
#
# Env (optional): BACKUP_DIR (default ./backups), BACKUP_RETENTION (default 14)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."  # project root (compose file location)

# Safe .env reader (the .env contains values like "7 days" that must not be eval'd)
get_env() { grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"'"; }

DB_USER="$(get_env POSTGRES_USER)"
DB_NAME="$(get_env POSTGRES_DB)"
SERVICE="timescaledb"
BACKUP_DIR="${BACKUP_DIR:-./backups}"
RETENTION="${BACKUP_RETENTION:-14}"

: "${DB_USER:?POSTGRES_USER not found in .env}"
: "${DB_NAME:?POSTGRES_DB not found in .env}"

mkdir -p "$BACKUP_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BACKUP_DIR/klines_${TS}.dump"

echo "[backup] $(date -u) dumping '$DB_NAME' -> $OUT"
# -Fc = custom (compressed) format, restorable with pg_restore
docker compose exec -T "$SERVICE" pg_dump -U "$DB_USER" -d "$DB_NAME" -Fc > "$OUT"

if [ ! -s "$OUT" ]; then
    echo "[backup] ERROR: dump is empty, removing." >&2
    rm -f "$OUT"
    exit 1
fi
echo "[backup] ok: $(du -h "$OUT" | cut -f1)"

# Retention: keep only the newest $RETENTION dumps
ls -1t "$BACKUP_DIR"/klines_*.dump 2>/dev/null | tail -n +"$((RETENTION + 1))" | while read -r old; do
    echo "[backup] pruning $old"
    rm -f "$old"
done
echo "[backup] retained newest $RETENTION dump(s) in $BACKUP_DIR"
