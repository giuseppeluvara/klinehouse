# PLAN — Confronto minimo Logistic vs Transformer per BTCUSDT

## 1. Obiettivo

Rispondere a una sola domanda: un Transformer encoder temporale compatto mostra
capacità predittiva fuori campione superiore a una regressione logistica usando
gli stessi campioni, feature, split e metriche?

Il Transformer viene eseguito incondizionatamente. Può essere bloccato soltanto
da leakage dimostrato, dati corrotti o assenti, oppure impossibilità tecnica
anche dopo la riduzione del batch. Una baseline negativa non è un motivo di
blocco. Questo file è la fonte unica dello stato durante l'implementazione.

## 2. Pulizia controllata

Commit di partenza: `892885799aa9d2a68aa22c0a6675008b909a897b`.

| Elemento | Decisione |
| --- | --- |
| Collector, database, volumi e 90 giorni raw `1s` | Conservare invariati |
| `PLAN.md` | Sostituire con questo piano |
| `scripts/backfill.py --from` | Conservare; era preesistente |
| Guardrail 15 GB e 20 GiB liberi | Conservare in forma minima |
| Worker pool e riscrittura HTTP del backfill | Ripristinare alla versione iniziale |
| `research/db.py` read-only | Conservare come opzione esplicita, non come default globale |
| `research/ml.py`, `ml_data.py`, `ml_models.py` | Rimuovere |
| Test ML attuali | Rimuovere e sostituire |
| Sezione ML aggiunta a `research/README.md` | Sostituire |
| `research/out/ml` | Eliminare come materiale contaminato e rigenerabile |
| Claim preesistenti del README | Fuori ambito; revisione separata |

Le primitive esistenti vengono recuperate soltanto se indispensabili al nuovo
confronto. Non viene preservata la struttura della pipeline bocciata.

## 3. Protocollo fisso

- Target unico: `BTCUSDT`.
- Contesto: BTCUSDT, ETHUSDT e SOLUSDT.
- Mercato: Binance Spot.
- Frequenza: `1m`.
- Lookback: 120 barre, pari a due ore.
- Orizzonte: 60 minuti.
- Anchor: ogni ora UTC.
- Copertura: tutta l'intersezione temporale disponibile prima di `T0`.

Si usano archivi Binance mensili per i mesi completi e giornalieri soltanto per
la coda, verificando e registrando ogni checksum. Non è richiesta continuità
globale: si escludono localmente soltanto gli anchor con dati incompleti. Il raw
`1s` viene conservato ma non entra in questo confronto.

Per un anchor `t`:

- input: barre con `open_time` in `[t-120m, t)`;
- ingresso teorico: open BTC della barra `t`;
- uscita: close BTC della barra `t+59m`;
- label: intervallo `[t, t+60m)`.

Gli intervalli label consecutivi non si sovrappongono.

Il log-return per asset e minuto `s` è
`log(close(s) / close(s-1))`. La sequenza contiene 120 ritorni; viene caricata e
verificata anche la barra di supporto `t-121m`, che non è un timestep del
modello. Un campione richiede 121 barre consecutive per tutti gli asset, 120
barre sincronizzate e 60 barre future BTC complete.

## 4. Target e feature

Il target, congelato prima di osservare i risultati, è:

```text
r(t) = log(BTC_close(t+59m) / BTC_open(t))
down  se r(t) <= -0,002
flat  se -0,002 < r(t) < +0,002
up    se r(t) >= +0,002
```

Per ogni asset e timestep si usano quattro feature: log-return close-to-close,
range `(high-low)/open`, `log1p(quote_volume)` e imbalance taker
`(2*taker_buy_base-volume)/volume`. Totale: 12 canali.

Entrambi i modelli ricevono lo stesso tensore `120x12`: la logistic lo usa
appiattito, il Transformer come sequenza. Lo scaler per canale viene stimato
soltanto sul training. I campioni incompleti vengono esclusi, non imputati.

## 5. Riferimenti e modelli

Il class-prior non addestrato di ciascun fold è la distribuzione delle classi
calcolata sul solo training e restituita come previsione costante. Misura la
capacità assoluta di logistic e Transformer.

Logistic congelata:

- `C=1.0`, `solver="lbfgs"`, `class_weight=None`;
- `max_iter=2000`, `tol=1e-4`, `random_state=7`;
- classificazione multinomiale, CPU e thread BLAS fissati.

Transformer congelato:

- input `120x12`, proiezione `12 -> 32`, encoding sinusoidale;
- due encoder layer pre-norm, `d_model=32`, quattro teste;
- feed-forward 64, dropout 0,1, mean pooling e head a tre classi;
- meno di 25.000 parametri;
- cross-entropy non pesata, AdamW `3e-4`, weight decay `1e-2`;
- batch 256, dieci epoche, clipping 1,0; seed 7, 17 e 29.

CPU deterministica è il backend primario. Il batch può scendere a 128 o 64
soltanto per memoria insufficiente, mai per accelerare la CPU. Un benchmark sul
solo training può autorizzare MPS se la CPU supera il budget totale di due ore,
senza leggere validation/holdout e senza cambiare modello, epoche o campioni.
Backend e limiti di riproducibilità vengono registrati.

## 6. Previsione e sequenza OOF primaria

Per ogni campione, la previsione Transformer primaria è
`p_ensemble = (p7 + p17 + p29) / 3`; la log-loss primaria viene calcolata su
questa probabilità. Non si sceglie il seed migliore. Loss individuali, media,
dispersione e metriche per seed restano misure secondarie di stabilità.

Le probabilità out-of-fold dei tre blocchi walk-forward vengono concatenate in
ordine cronologico e ordinate stabilmente per timestamp e sample ID. La
sequenza OOF concatenata è la sorgente primaria per metriche aggregate,
differenze di log-loss per campione, bootstrap accoppiato settimanale e
conclusione. I blocchi bootstrap non attraversano i confini dei fold. I
risultati dei singoli fold restano metriche secondarie.

Confronti separati:

- relativo: log-loss ensemble Transformer meno log-loss logistic;
- assoluto: logistic meno class-prior ed ensemble Transformer meno class-prior.

Il Transformer è superiore soltanto se la differenza OOF media è negativa, il
limite superiore del CI bootstrap 95% è sotto zero e il segno è favorevole in
almeno due fold su tre. Non sono inclusi backtest, costi, soglie operative o
sweep economici.

## 7. Split, purging ed embargo

Si usano almeno 24 mesi iniziali di training e tre validation block consecutivi
da sei mesi, con training expanding e sample ID identici per tutti i modelli.

Per ogni sample `i` si costruiscono gli intervalli effettivi:

```text
input(i)   = [anchor(i)-120m, anchor(i))
label(i)   = [anchor(i), anchor(i)+60m)
support(i) = input(i) unione label(i)
```

Per il fold validation `V`, l'intervallo protetto va dal primo timestamp di
input validation fino a 120 minuti dopo l'ultimo `label_end` validation. Un
sample training è ammesso soltanto se il suo supporto non interseca questo
intervallo e il suo anchor appartiene al training expanding. I test verificano
gli estremi temporali reali e ogni intersezione, non una distanza nominale.

## 8. Training finale e freeze pre-`T0`

Dopo la validation OOF, senza modificare il protocollo, vengono selezionati
tutti e soli i campioni pre-`T0` i cui input e label terminano completamente
prima del cutoff. Gli esatti sample ID vengono ordinati, registrati e congelati.

Su questo identico insieme, una sola volta:

1. si stima lo scaler finale;
2. si calcola il class-prior finale;
3. si addestra la logistic finale;
4. si addestrano i Transformer finali con seed 7, 17 e 29.

Non si effettua selezione ulteriore né si riutilizzano le predizioni OOF come
dati di training.

Prima di `T0` si congelano tramite hash: commit pulito, configurazione, cutoff
sorgente, checksum, dataset, sample ID finali e per fold, scaler, class-prior,
modelli, backend e versioni delle dipendenze. Un unico `freeze.json` registra il
bundle. `T0` è il primo giorno UTC completo successivo alla verifica del bundle;
`T1 = T0 + 180 giorni`. Fino a `T1`: `final test: pending`.

## 9. Procedura unica a `T1`

Quando i dati arrivano almeno a `T1+60m`, si verifica che non esista già un
record di apertura, si caricano e verificano `freeze.json`, scaler finale,
class-prior finale, logistic finale e i tre Transformer finali. Non si
riaddestra né aggiorna nulla e non si ricalcolano le frequenze sull'holdout.

Si costruiscono una sola volta i sample `[T0,T1)`, si applicano scaler e modelli,
si calcola l'ensemble dei tre seed e si registrano una sola volta metriche,
timestamp e hash del test. Il risultato non è sovrascrivibile. Qualunque analisi
successiva viene marcata `post-hoc`.

## 10. Piano operativo

| Milestone | Stato iniziale | Verifica |
| --- | --- | --- |
| M1 — Piano e pulizia | complete | Riferimenti orfani assenti; raw e servizi invariati |
| M2 — Storico `1m` | complete | Checksum verificati; gap gestiti localmente |
| M3 — Dataset causale | complete | Rebuild deterministico e fingerprint completo |
| M4 — Test sostanziali | complete | Causalità, coda e intervalli reali verificati |
| M5 — Prior, logistic e Transformer | in progress | Stessi sample e fold; tutti i seed eseguiti |
| M6 — Report validation | pending | Metriche primarie sulla sequenza OOF concatenata |
| M7 — Training finale e freeze | pending | Un solo training sui sample ID completi pre-`T0` |
| M8 — Holdout prospettico | pending fino a `T1` | Unica apertura con bundle congelato |

## 11. Superficie software

Le stime sono limiti massimi, non obiettivi.

| File | Azione | Limite |
| --- | --- | ---: |
| `PLAN.md` | Sostituzione completa | Piano compatto |
| Tre moduli `research/ml*` | Rimozione | -980 righe circa |
| Test ML attuali | Rimozione | -119 righe |
| `research/sequence_experiment.py` | Nuovo modulo canonico | Massimo 500 righe |
| `tests/test_sequence_experiment.py` | Test sostanziali | Massimo 200 righe |
| `research/README.md` | Sostituzione sezione ML | Circa 15 righe |
| `research/requirements.txt` | Dipendenze effettive | 2-3 modifiche |
| `research/db.py` | Read-only opt-in | Circa 5 righe |
| `scripts/backfill.py` | Ripristino più guardrail | Circa 20 righe nette |
| `README.md` | Sola documentazione backfill | Circa 10 righe |

Se il codice o i test raggiungono i limiti, devono essere semplificati. I claim
preesistenti del README restano fuori ambito.

## 12. Test e completamento

I test devono coprire: prefix-invariance, barra `t-121m`, gap locali, coda
futura, target non sovrapposti, intervalli effettivi di purge/embargo, scaler
train-only, identità dei sample fra modelli, concatenazione OOF, rebuild hash,
training dei tre seed, ensemble completo, class-prior congelato e apertura
finale non sovrascrivibile.

Il confronto è completo quando prior, logistic e Transformer producono
predizioni sugli stessi fold; tutti i seed sono eseguiti; metriche e bootstrap
primari usano la sequenza OOF; il training finale usa una sola volta tutti e
soli i sample ID completi pre-`T0`; il bundle è congelato; nessuna valutazione
economica è inclusa; il test resta pending fino a `T1`.
