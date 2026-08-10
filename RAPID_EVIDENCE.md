# RAPID EVIDENCE — protocollo congelato

## Stato e separazione dal protocollo principale

Protocollo congelato il `2026-08-10T09:00:59Z`. `P0` è la prima ora UTC
completa successiva: `2026-08-10T10:00:00Z`.

Questo sprint è separato dal test prospettico originale. Il commit
`1a4cd9c723d94e5280c5532b5a56c24d5ade027b`, il tag locale
`sequence-t0-2026-08-08` e il bundle originale restano immutati. Lo sprint usa
soltanto dati e label antecedenti al vecchio `T0` per ablation e analisi OOF.
Il periodo `T0–P0` è escluso dal pilot.

L'apertura di risultati intermedi renderà scorretto descrivere l'intero
intervallo `T0–T1` come un singolo holdout mai osservato. Il futuro viene quindi
suddiviso ex ante in finestre indipendenti.

## Audit del freeze

L'audit read-only ha verificato commit, 234 checksum sorgente, 12 hash bundle,
dataset, OOF, report, sample ID, ordinamento, coda destra, split, prior, scaler
finale, logistic, ensemble e bootstrap. Non esistono file di apertura del test.

Due caveat non materiali vengono conservati, non corretti:

- la serializzazione float32 produce differenze massime di `2,02e-7` nelle
  loss ricostruite e `7,95e-8` nella media delle probabilità salvate;
- la barra ausiliaria `t-121m` non compare nell'intervallo nominale di purge e
  condivide un minuto con un solo sample train per fold. Sono tre sample su
  training fold da 39.318–48.078 righe; il difetto non è plausibilmente capace
  di spiegare il delta OOF di `-0,04291` e non autorizza a cambiare i fold.

## Domande e ipotesi

1. Ordine: lo shuffle temporale peggiora la log-loss con CI interamente sopra
   zero se l'ordine dei 120 step contribuisce realmente.
2. Non linearità: un MLP equivalente o migliore indica che aggregati e non
   linearità spiegano il vantaggio; un MLP significativamente peggiore sostiene
   la rappresentazione sequenziale.
3. Contesto: BTC-only significativamente peggiore indica informazione utile in
   ETH/SOL; equivalenza indica che il contesto non è necessario qui.
4. Economia: si misura, senza tuning, se le probabilità OOF superano un costo
   provvisorio realistico.
5. Prospettiva: i modelli congelati devono produrre eventi orari append-only
   senza alcun percorso di fit.

## Dati, fold e training comuni

- Target, tensore sorgente, 52.424 sample ID, tre fold e 13.104 validation ID
  sono quelli congelati.
- Target: log-return BTC `open(t) → close(t+59m)`; classi a `±0,002`.
- Metrica primaria: log-loss dell'ensemble, ottenuto mediando per campione le
  probabilità dei seed `7`, `17`, `29` prima della loss.
- Preprocessing stimato esclusivamente sul training del fold.
- PyTorch MPS, batch 256, 10 epoche, cross-entropy non pesata, AdamW
  `lr=3e-4`, weight decay `1e-2`, clipping `1,0`, dropout `0,1`.
- Nessun tuning, early stopping, grid search, seed selection o prova extra.

## Tre ablation, esattamente

### A — Transformer con ordine distrutto

Architettura e training originali. Shuffle seed unico `20260810`. Per ogni
sample ID si inizializza `SeedSequence([20260810, low32(id), high32(id)])` e si
applica la relativa `default_rng(...).permutation(120)` ai timestep. La stessa
funzione opera su train e validation; conserva il multiset e genera una
permutazione sample-specific.

### B — MLP su 48 aggregati

Per ciascuno dei 12 canali, in ordine, si calcolano soltanto ultimo valore,
media, deviazione standard population (`ddof=0`) e pendenza OLS rispetto agli
indici `0..119`. Gli input 48-dimensionali sono standardizzati per colonna sul
solo training. Architettura `48 → 128 → 64 → 3`, GELU e dropout `0,1` dopo i
primi due layer; mean pooling non applicabile. Training comune invariato.

### C — Transformer BTC-only

Si usano i canali originali `0..3` di BTC. Scaler per canale sul solo training;
proiezione input `4 → 32`. Encoding, encoder, pooling, head e training restano
identici all'originale.

## Confronto statistico

Ogni ablation viene confrontata con le probabilità OOF congelate del
Transformer originale sugli stessi ID. Si riportano log-loss, differenza di
loss per sample, fold secondari e block bootstrap settimanale accoppiato con
10.000 draw e seed 7; i blocchi non attraversano i fold.

- shuffled peggiore con CI 95% interamente sopra zero: contributo dell'ordine;
- MLP equivalente/migliore: vantaggio soprattutto non lineare/aggregato;
- MLP con CI interamente sopra zero: evidenza aggiuntiva sequenziale;
- BTC-only con CI interamente sopra zero: contributo ETH/SOL;
- BTC-only con CI comprendente zero: contesto non necessario dimostrabile.

## Diagnostica OOF descrittiva

Per prior, logistic e Transformer originali si calcolano log-loss per fold e
mese, loss per classe reale, multiclass Brier `mean(sum((p-y_onehot)^2))`, ECE
su dieci bin fissi `[0,0.1), …, [0.9,1]`, tabella di calibrazione, balanced
accuracy, macro-F1, quantili della confidenza massima e rendimento futuro nei
medesimi dieci bin fissi. Le settimane sono definite da blocchi UTC di sette
giorni che non attraversano i fold. Si riporta la percentuale in cui il
Transformer ha loss media inferiore alla logistic. Nessun risultato modifica
modello, target o soglie.

## Analisi economica esplorativa congelata

Venue/prodotto/fee tier non sono configurati in modo inequivocabile. Si usa un
unico costo provvisorio di **20 bps round-trip** (`0,002` log-return per trade),
che non certifica né rappresenta le commissioni Kraken.

Per ogni fold, `mu_down`, `mu_flat`, `mu_up` sono le medie dei return continui
calcolate soltanto sui sample ID del training passato. Per logistic e
Transformer:

```text
expected_return = p_down*mu_down + p_flat*mu_flat + p_up*mu_up
long  se expected_return > 0,002
short se expected_return < -0,002
flat  altrimenti
```

Gross PnL per trade: `position*realized_log_return`; net PnL: gross meno `0,002`
per posizione non nulla. Turnover è `2*trade_count` unità one-way. Max drawdown
è sul cumulato netto cronologico. Break-even round-trip è
`sum(gross_pnl)/trade_count`, espresso anche in bps. Non si eseguono sweep,
confidence cutoff o tuning e non si afferma redditività futura.

## Shadow trading append-only

Il runner carica i soli scaler, class-prior, logistic e tre Transformer del
bundle originale, verificandone gli hash. Non contiene né importa funzioni di
fit. Le medie di classe finali, calcolate una volta sui 52.424 ID pre-`T0`, sono
un parametro descrittivo congelato del rapid sprint, non un modello aggiornato.

`python -m research.rapid_evidence shadow-once` usa esclusivamente raw 1s già
chiusi per aggregare 121 minuti causali dei tre asset fino all'ultimo anchor
orario completo `>=P0`. Scrive un evento per sample ID in
`research/out/sequence/rapid/events.jsonl`, con probabilità, expected return,
decisione, hash bundle e ID. Un ID già presente non viene riscritto.

Quando sono disponibili i 60 minuti futuri, una successiva invocazione aggiunge
la label a `maturations.jsonl`; l'evento originale non viene modificato. Anche
la maturazione è idempotente. Nessuna metrica prospettica viene stampata o
aggregata prima dell'apertura prevista.

Scheduling suggerito, senza daemon proprietario:

```cron
5 * * * * cd /Users/dev-luv/Sites/klinehouse && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m research.rapid_evidence shadow-once >> research/out/sequence/rapid/shadow.log 2>&1
```

## Finestre prospettiche congelate

| Finestra | Inizio UTC | Fine UTC | Apertura |
| --- | --- | --- | --- |
| Pilot | 2026-08-10 10:00 | 2026-09-09 10:00 | Solo dopo maturazione ultima label |
| Conferma | 2026-09-09 10:00 | 2026-11-08 10:00 | Solo se versione originale invariata |
| Replicazione lunga | 2026-11-08 10:00 | 2027-02-04 00:00 | Alla fine del vecchio `T1` |

Nei primi sette giorni si controllano soltanto funzionamento, gap, duplicati,
timestamp e maturazioni. Nessuna performance viene letta prima del termine del
pilot. Modifiche dopo il pilot richiedono un nuovo `T0`; analisi successive a
ogni apertura sono `post-hoc`.

## Hash congelati rilevanti

| Artefatto | SHA-256 |
| --- | --- |
| `freeze.json` | `0995cbdad98a20947f09cd243a483d963287b933b1c09804bbf90922d50def7d` |
| `oof.parquet` | `c43b627bf667e09b7810af6d9be03a5bb27be69bfec4b8004860269e8701e227` |
| `report.json` | `21dabe4ac11ac93c0718ae4a177697fdb455d12cf85dced476353531ea1deb4c` |
| Dataset | `6b639f3580a4bff9a310df500cc09f47a16f459be81134ffb851b53d3fc3fe02` |
| Sample ID finali | `8d6935ea26f7bf0b9ad5a850a0541ea8eece8ebd10aaa65fb2dc3405defabf07` |
| `PLAN.md` | `7c8e866e2da5f674bd2007a80f25c50b1b5e86dabac544e2ee11490c9ed559f3` |
| `sequence_experiment.py` | `391a3f403180d13b3cceec58c01b83c41f77be4dbb18f559ff779be1c7070fcb` |
| `test_sequence_experiment.py` | `9dafed47ed391640d980296665cd209bc73561a5520d4f47fbfacf9afbc1ee2d` |
