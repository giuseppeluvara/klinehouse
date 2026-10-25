# Klinehouse — storage e pilot Timestral, 25 ottobre 2026

Il lettore è stato corretto e verificato su dati reali. Timestral 5M ha completato
144 previsioni per ciascuna delle due varianti, con protocollo registrato prima dei
risultati. Il confronto **non dimostra un vantaggio operativo**.

## Ambiente e perimetro

Mac mini M4, 16 GiB RAM, macOS 26.5.1 ARM, Python 3.13.5. Su `main` è stato
preservato il commit locale `c056855` integrando `5e17db1`, senza riscrivere la
cronologia. `.env`, protocolli precedenti e i due file utente `encoder_backtest`
sono invariati; nessun vecchio holdout è stato aperto.

PostgreSQL su localhost:5433 rifiuta la connessione; Docker e gli archivi precedenti
non sono presenti. **Non verificati:** migrazione, dimensioni, copertura BTC/ETH/SOL,
politiche e simulazione della potatura del DB di produzione. Nessun `--apply`
eseguito sul database dell'utente.

## Storage: misure reali

Fonte: [dataset ufficiale Binance](https://github.com/binance/binance-public-data),
BTCUSDT Spot 1s, `[2026-07-01, 2026-07-08)` UTC, prima del vecchio holdout.
**604.800 candele**, 7 giorni completi, 10.080 minuti validi. ZIP verificati con
checksum ufficiali; Parquet riletti integralmente. Sono candele OHLCV, non trade tick.

| Misura | Risultato |
|---|---:|
| CSV sorgenti, non estratti | 88.096.035 byte |
| ZIP conservati | 15.746.428 byte |
| Parquet Zstandard, 7 file | **22.871.811 byte** |
| Download | 18,26 s |
| Esportazione e verifica | 28,88 s |
| Comando completo / picco RSS | 52,03 s / 205 MiB |

Conservati prezzi/volumi float64 come nello schema DB, conteggi interi e timestamp
UTC in microsecondi. Creata una copia indipendente verificata dei 7 file, sullo
**stesso SSD**: non protegge dal guasto del disco e non è un backup del DB reale.

Letture in processi distinti, due prove, batch massimo 8.192 righe, ordine verificato:

| Query | Righe | Tempo | Picco RSS |
|---|---:|---:|---:|
| 1 giorno, tutte le colonne | 86.400 | 0,578–0,582 s | 220 MiB |
| 7 giorni, timestamp e close | 604.800 | 0,849–0,854 s | 314–319 MiB |
| 7 giorni a 1m, 6 colonne | 10.080 | 0,314–0,317 s | 275–276 MiB |

Cache del sistema operativo non svuotata. Il limite DuckDB di 256 MB non è la
RAM totale. Nessuna misura comparativa del DB, nessuna cache derivata aggiunta.

Correzioni: backfill che preserva tutte le chiavi esistenti, valori e percorsi
validati, esclusione delle candele non ancora chiuse, risorse rilasciate su
interruzione. Il primo run si è fermato **prima di produrre previsioni** per un
riordinamento dei batch Arrow in DuckDB 1.5; corretto e coperto da una regressione
su 172.800 righe. Il tentativo fallito è conservato, senza cambiare il protocollo
statistico della nuova esecuzione.

Il dry-run della potatura ora confronta realmente ogni simbolo/colonna del DB con
archivio e backup, in sola lettura, e blocca retention/refresh incompatibili.
Backup atomico e modalità applicativa con lock sono descritti in
[storage.md](../../docs/storage.md).

## Timestral: risultato esplorativo

[Codice Mistral](https://github.com/mistralai/timestral) fissato a
`a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`; checkpoint
[Timestral-5M](https://huggingface.co/mistralai/Timestral-5M/tree/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f)
fissato a `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` (20,84 MB), hash verificati.
CPU, 2 thread, zero-shot. Codice Apache 2.0; pesi Modified MIT con condizione
commerciale, dettagli e licenza in [timestral.md](../../docs/timestral.md).

256 log-prezzi a 1m; target log-rendimento a 60 minuti; anchor non sovrapposti.
19 campioni iniziali per calibrare le baseline, poi 144 anchor comuni in sei
blocchi giornalieri walk-forward, dal 2 al 7 luglio. Solo esiti già maturati per
la calibrazione. Covariate: volatilità interna al minuto, intensità degli scambi,
imbalance taker. Nessun riempimento dei gap, tuning o informazione futura.

Errori e pinball in punti base di **log-rendimento**, più basso è meglio:

| Modello | MAE | RMSE | Pinball | Direzione | Copertura 90% |
|---|---:|---:|---:|---:|---:|
| Prezzo invariato | 28,47 | 41,20 | 8,11 | neutrale | 93,06% |
| Drift locale | 33,04 | 47,22 | 9,20 | 52,08% | 92,36% |
| Timestral prezzo | 28,21 | 41,47 | 7,90 | 50,69% | 86,11% |
| Timestral + covariate | 28,51 | 41,99 | 8,02 | 50,69% | 86,11% |

Prezzo: MAE migliore dello 0,92%, RMSE peggiore; MAE migliore soltanto in 2/6 giorni.
Le covariate non aiutano complessivamente. Copertura inferiore al 90% nominale,
campione piccolo; nessun quantile incrociato. La direzione zero della baseline
invariata nel JSON indica neutralità, non previsioni tutte sbagliate.

Blocco strumentato: 16,40 s, esclusa la verifica iniziale dell'archivio;
**picco RSS 748 MiB**. Inferenza mediana: prezzo 9,20 ms, covariate 15,64 ms;
non include ritardo di arrivo dati né garantisce un prezzo eseguibile.

I [dati di training dichiarati](https://mistral.ai/news/timestral/) non hanno un
inventario/cutoff completo: contaminazione storica non escludibile. Nessun PnL,
ordine o redditività dichiarata. Comandi prospettici implementati, emissione entro
60 s e scoring separato; **nessun test prospettico eseguito**.

## Verifiche, evidenze e prossimo passo

67 test mirati locali passati, 8 integrazioni saltate senza DB; Ruff superato.
34 test legacy passano; 2 richiedono bundle/dataset originali assenti e non sono
stati alterati. **CI superata: 75 test, incluse 8 integrazioni Timescale reali**;
lint e import superati. La CI isolata prova anche politiche, backup, dati correnti,
aggregati e blocco dei backfill: commit ed esito in `ci.json`.

Questa cartella contiene protocollo, provenance, benchmark, tutte le previsioni,
quantili, esiti, metriche giornaliere e ambiente; `sha256.json` verifica le evidenze.
ZIP/Parquet/backup/pesi restano in `research/out/timestral-pilot-20261025/`, fuori
Git. Dipendenze osservate in `requirements-timestral-repro.txt`; comandi completi
nella [guida](../../docs/timestral.md). Ripetere in una nuova directory, ad esempio
`run-reproduction`; il protocollo salvato conserva i percorsi originali.

SSD: 73,78 → 73,60 GiB liberi (circa 0,18 GiB di variazione del sistema).
Ambiente aggiunto circa 53 MiB, dati/modello/backup circa 85 MiB; nessuna dipendenza
protetta eliminata.

Sul Mac che ospita realmente PostgreSQL, dalla root del repository:

```bash
python -m pip install -r requirements-storage.txt
python scripts/audit_storage.py --root archive --output research/out/storage-audit.json
python scripts/archive_market.py --root archive sync
python scripts/archive_market.py --root archive verify
python scripts/backup_archive.py --root archive --destination /percorso/separato/backup-nuovo
python scripts/prune_archive.py --root archive --backup-root /percorso/separato/backup-nuovo
```

Prossimo passo: verificare la migrazione reale e raccogliere previsioni prospettiche
con questo protocollo, prima di considerare una regola operativa.
