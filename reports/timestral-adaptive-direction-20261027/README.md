# Probabilità adattive: correzione misurabile, direzione ancora instabile

Sono stati implementati ed eseguiti due approcci adattivi sui medesimi nove
confronti: BTC, ETH e SOL, gennaio–luglio di ciascuno degli anni 2024, 2025 e 2026.
**Nessuno supera tutti i criteri prefissati.** Entrambi passano cinque controlli
direzionali su nove; il modello originale ne passava sei. Gli holdout dal
1° agosto 2026 in avanti non sono stati aperti.

Il risultato utile è una diagnosi più precisa: ridurre l'eccesso di sicurezza
è possibile in alcuni periodi, ma non basta a ricostruire un segnale direzionale
stabile. Inoltre, gran parte del miglioramento della log-loss della calibrazione
riguarda la probabilità di prezzo **invariato**, non rialzo contro ribasso.

## Due prove registrate prima dell'esecuzione

Il primo metodo corregge ogni giorno le probabilità del modello lineare comune.
Usa solo previsioni storiche fuori campione e i loro esiti già maturati entro
la mezzanotte UTC: finestra di 28 giorni, peso dimezzato ogni sette giorni.
Stima separatamente la prudenza sulla direzione e uno spostamento della
probabilità di prezzo invariato. La prudenza può solo ridurre il rapporto di
probabilità tra rialzo e ribasso; non può invertirne l'ordine.

In formula, la probabilità condizionata di rialzo è
`sigmoid(alpha * log(p_up / p_down))`, con `0 <= alpha <= 1`;
la probabilità invariata è `sigmoid(logit(p_flat) + offset)`, con
`-3 <= offset <= 3`. La ricombinazione produce tre probabilità che sommano a uno.
La stima minimizza la log-loss pesata con una piccola penalità prefissata verso
la trasformazione identità. Si tratta di una calibrazione personalizzata:
la [documentazione scikit-learn](https://scikit-learn.org/stable/modules/calibration.html)
descrive il principio della calibrazione su previsioni separate dal fitting e
i limiti della semplice modifica della temperatura.

Il secondo metodo, registrato dopo il risultato del primo, cambia ogni giorno
i pesi di quattro previsioni già congelate: modello lineare comune, alberi,
reversal e frequenze storiche. I pesi sono non negativi, sommano a uno e
minimizzano la log-loss pesata con regolarizzazione entropica prefissata.
Finestra, decadimento, ritardo degli esiti e calendario sono uguali al primo
metodo. Questo approccio può cambiare anche la direzione scelta.

Per entrambi:

- la regola e gli iperparametri sono identici per tutti gli asset e gli anni;
- i parametri sono specifici dell'asset, senza trasferire la calibrazione da un
  anno all'altro;
- i primi sette giorni usano il modello originale e **restano nelle metriche**;
- il controllo fisso conserva la prima correzione, stimata l'8 gennaio;
- il controllo originale conserva le probabilità senza correzione;
- soglia di selezione 52%, copertura minima 5% e criteri direzionali precedenti
  rimangono invariati. Non sono stati cercati altri valori dopo i risultati.

I protocolli sono in [protocol.json](protocol.json) e
[mixture-refined/protocol.json](mixture-refined/protocol.json), con hash di
codice, dati e cronologia. Questi periodi erano già stati esplorati: sono nuove
prove retrospettive, non una validazione prospettica o un nuovo holdout.

## Risultati direzionali su tutti i periodi

Accuratezza e copertura sono percentuali. L'accuratezza riguarda solo i casi
selezionati; le selezioni cambiano tra metodi. Il criterio richiede anche
qualità delle probabilità, intervalli favorevoli, entrambe le direzioni e
stabilità mensile: un'accuratezza sopra 52% da sola non basta.

| Periodo / asset | Originale: criterio | Calibrazione: accuratezza / copertura | Criterio | Miscela: accuratezza / copertura | Criterio |
|---|---|---:|---|---:|---|
| 2024 BTC | Superato | 52,91 / 23,34 | Superato | 53,14 / 24,53 | Superato |
| 2024 ETH | Non superato | 52,76 / 17,32 | Non superato | 53,05 / 10,22 | Non superato |
| 2024 SOL | Non superato | 51,30 / 9,91 | Non superato | 52,35 / 7,67 | Non superato |
| 2025 BTC | Non superato | 51,63 / 12,32 | Non superato | 52,31 / 15,33 | Non superato |
| 2025 ETH | Superato | 54,01 / 23,46 | Superato | 54,88 / 13,72 | Superato |
| 2025 SOL | Superato | 52,70 / 7,97 | Non superato | 54,41 / 4,20 | Non superato |
| 2026 BTC | Superato | 53,35 / 32,58 | Superato | 53,56 / 32,09 | Superato |
| 2026 ETH | Superato | 53,87 / 39,22 | Superato | 54,28 / 35,20 | Superato |
| 2026 SOL | Superato | 53,09 / 13,68 | Superato | 52,97 / 11,04 | Superato |

La miscela su SOL 2025 illustra il rischio di guardare solo l'accuratezza:
54,41% è accompagnato da copertura 4,20%, inferiore al minimo registrato.
Il metodo rimane respinto su quel confronto.

## Migliorano davvero le probabilità?

Il confronto aggiuntivo usa **tutti gli stessi anchor**, comprese astensioni e
avvio, con differenze appaiate di Brier e log-loss rispetto al modello originale
e, separatamente, rispetto al controllo fisso. Gli intervalli sono bootstrap
a blocchi mobili di sette giorni, 4.000 repliche, seed fisso; livello 99,444%
per correggere nove confronti asset/anno per ciascuna metrica e ciascun controllo.
Non è una correzione per l'intera storia di ricerca o per tutti gli endpoint.
Gli intervalli sono condizionati alle previsioni salvate, senza rifitting in
ciascuna replica bootstrap.

| Metodo adattivo | Log-loss migliore dell'originale con intervallo positivo | Brier e log-loss entrambi migliori dell'originale | Brier e log-loss entrambi migliori del controllo fisso |
|---|---:|---:|---:|
| Calibrazione | 5 / 9 | 0 / 9 | 1 / 9 |
| Miscela | 5 / 9 | 1 / 9 | 1 / 9 |

L'unico successo congiunto della miscela contro l'originale è SOL 2026:
guadagno Brier assoluto con intervallo `[0,0000448; 0,0005671]` e guadagno
log-loss `[0,0006827; 0,0022006]`. Non supera però il confronto con il controllo
fisso sullo stesso caso. Le singole superiorità al controllo fisso sono SOL 2024
per la calibrazione ed ETH 2025 per la miscela. Non dimostrano un vantaggio
generale dell'aggiornamento giornaliero.

I risultati completi, inclusi i controlli e i mesi, sono in
[results.json](results.json) e [mixture-refined/results.json](mixture-refined/results.json).

## Che cosa spiega il fallimento residuo

Una scomposizione **descrittiva, eseguita dopo la valutazione**, separa la
log-loss sul prezzo invariato dalla log-loss della direzione condizionata a un
movimento. Le due componenti sommano alla log-loss multiclass. In tutti e cinque
i casi in cui la calibrazione migliora significativamente la log-loss totale,
il guadagno medio sulla componente invariata è maggiore di quello direzionale.
Per esempio, su ETH 2025 la componente invariata migliora di circa 0,000585,
mentre quella direzionale peggiora di circa 0,000027.

L'eccesso medio di sicurezza, misurato sugli **stessi casi e sulla stessa classe
direzionale selezionati dal modello originale**, passa:

| Periodo / asset | Originale, punti percentuali | Dopo calibrazione, punti percentuali |
|---|---:|---:|
| SOL 2024 | +1,56 | +0,41 |
| BTC 2025 | +1,26 | −0,07 |
| BTC 2026 | +0,89 | +0,04 |

Il segno negativo indica sottostima media. Queste medie non provano calibrazione
in ogni fascia di probabilità e non sono nuovi criteri di successo.
La riduzione dell'eccesso di sicurezza nei primi due casi non li fa superare
il criterio direzionale. Le diagnostiche sono conservate e ricostruite
indipendentemente in [diagnostics.json](diagnostics.json).

L'evidenza è quindi compatibile con due problemi distinti: probabilità che
richiedono correzione e segnale direzionale debole o instabile. Non identifica
una causa unica di tutti i fallimenti. Il prossimo esperimento utile dovrebbe
verificare se informazioni aggiuntive migliorano la componente rialzo/ribasso,
mantenendo esplicito il confronto con il prezzo invariato; la sola calibrazione
non ha fornito questa prova.

## Correzione numerica e verifiche

Il primo tentativo della miscela si è fermato prima di produrre metriche di
coorte: SLSQP dichiarava successo, ma il residuo KKT era `2,98e-6`, sopra il
limite `2e-6`. Il codice e il protocollo originari restano conservati, con
[failure.json](mixture/failure.json). Una seconda registrazione ha cambiato
solo precisione e numero massimo di iterazioni del solver (`ftol=1e-15`,
`maxiter=500`), lasciando invariati obiettivo, soglia KKT e disegno statistico.

I verificatori non importano modelli o runner di produzione. Ricostruiscono
membership temporale, esclusione degli esiti immaturi, pesi, probabilità,
metriche, intervalli, diagnostiche e criteri. Verificano le condizioni di
ottimalità convessa senza rieseguire l'ottimizzazione: **1.848 aggiornamenti
per metodo, 3.696 complessivi**. Residui massimi circa `9,85e-15` per la
calibrazione e `1,29e-7` per la miscela.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python scripts/verify_adaptive_direction.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python scripts/verify_adaptive_mixture.py
```

Le ricevute sono [verification.json](verification.json) e
[mixture-refined/verification.json](mixture-refined/verification.json).
I test sintetici controllano cambi di regime, passaggio tra esperti, simmetria,
ritardo delle etichette, dati futuri e confine protetto. La suite locale passa
**253 test**, con **8 test PostgreSQL saltati** per assenza del database locale.
La CI esegue anche questi test e gli audit degli esperimenti precedenti.

Per ripetere fitting e valutazione, creare destinazioni nuove: i comandi di
registrazione ed esecuzione rifiutano la sovrascrittura degli artefatti.

```bash
.venv-timestral/bin/python scripts/run_adaptive_direction.py register --root research/out/adaptive-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python scripts/run_adaptive_direction.py run --root research/out/adaptive-replay
.venv-timestral/bin/python scripts/run_adaptive_mixture_refined.py register --root research/out/adaptive-replay/mixture-refined
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python scripts/run_adaptive_mixture_refined.py run --root research/out/adaptive-replay/mixture-refined
```

I comandi di audit ordinari verificano il bundle pubblicato completo, incluse le
diagnostiche e il tentativo numerico conservato; i replay sopra riproducono
fitting, previsioni e metriche, senza ricreare automaticamente queste ricevute.

Nessun download, modello aggiuntivo o nuova installazione. Il nuovo bundle
occupa circa **50,2 MiB**, esclusi gli oggetti Git. Fonti, `.env`, file locali
dell'utente e precedenti esperimenti restano preservati. Nessun ordine reale,
simulazione dei costi o validazione prospettica è stato eseguito. Il problema
della previsione direzionale generale rimane aperto.
