# Direzione a breve termine: un candidato selettivo su ETH

Il nuovo approccio trova un risultato storico positivo su **ETHUSDT**, sul
movimento dei cinque minuti successivi a **un minuto di ritardo**. Prevede la
direzione solo nel **24,37%** dei casi e indovina il **52,74%** di questi. Supera il
criterio individuale fissato prima di acquisire i dati ETH/SOL. **BTC e SOL non
lo superano; il criterio congiunto di trasferimento resta falso.**

È un candidato direzionale selettivo verificato sui dati storici consentiti,
non una soluzione generale o una prova prospettica. Il risultato sulla volatilità
del precedente esperimento non viene utilizzato come prova della direzione.

## Risultato principale

Valutazione gennaio–luglio 2026; addestramento mensile sui 90 giorni precedenti.
Il denominatore è di 61.055 anchor per asset. Gli esiti esattamente invariati
rimangono nel test e contano come errori quando viene prevista una direzione.

| Asset | Previsioni selezionate | Copertura | Direzioni corrette | Intervallo storico 95% | Guadagno Brier sulle selezionate | Criterio individuale |
|---|---:|---:|---:|---:|---:|---|
| BTC | 9.710 | 15,90% | 51,62% | 50,73–52,73% | +0,086% | Non superato |
| ETH | 14.881 | 24,37% | **52,74%** | **51,98–53,68%** | **+0,337%** | **Superato** |
| SOL | 3.406 | 5,58% | 51,20% | 49,86–53,28% | +0,274% | Non superato |

Il guadagno Brier è relativo alla frequenza delle tre classi stimata sul passato,
valutata sugli stessi casi selezionati. È un miglioramento percentuale dello
score, non punti percentuali di accuratezza. Per ETH il suo intervallo è
**+0,131–+0,579%**; la log-loss scende da 0,709770 a 0,709069. Il vantaggio Brier
è positivo in sei dei sette mesi. Sul campione ETH selezionato, la baseline
di frequenza indovina il 49,73%.

Su ETH le previsioni di rialzo sono corrette nel **52,73%** dei 7.279 casi;
quelle di ribasso nel **52,75%** dei 7.602 casi. La probabilità media assegnata
alla direzione scelta è 53,22%, rispetto al 52,74% osservato: lo score favorevole
non dimostra una calibrazione perfetta. La regola aggregata di ritorno verso il
prezzo precedente sceglie le stesse direzioni nei casi selezionati; i regimi
servono alla selezione e alla stima delle probabilità.

## Ipotesi, verifiche e limiti della diagnosi

1. **Informazione persa aggregando a ore.** Ho ricostruito 480.960 minuti da
   28.857.600 candele BTC al secondo già presenti, dal settembre 2025 al luglio
   2026. Ho confrontato frequenze incondizionate, regimi di prezzo/attività,
   aggiunta del flusso taker del minuto, e aggiunta degli ultimi dieci secondi.
   Orizzonti 1, 5 e 15 minuti; ritardi 0 e 1 minuto. Tutti i sei confronti sono
   conservati. Nel test principale a cinque minuti con ritardo, il modello
   completo peggiora il Brier dello **0,161%** rispetto alla frequenza e dello
   **0,176%** rispetto al solo prezzo. Il dettaglio di flusso non risolve il
   problema in questa rappresentazione.
2. **Troppi parametri rispetto a un segnale debole.** Dopo quel risultato ho
   registrato un distinto modello simmetrico: raggruppa rialzi e ribassi secondo
   la possibilità di inversione o continuazione. Usa dodici regimi, senza rete
   neurale, ottimizzazione o ricerca di iperparametri. Su BTC trova una piccola
   capacità di classificazione, ma sovrastima la fiducia e non supera il criterio
   probabilistico. I risultati BTC sono esplicitamente esplorativi: gli stessi
   periodi di prezzo erano già stati esaminati.
3. **Trasferibilità senza ulteriori aggiustamenti.** Ho congelato metodo,
   selezione e criteri, poi acquisito i minuti nativi ETH e SOL consentiti.
   L'addestramento resta specifico per asset e usa soltanto il passato. ETH
   supera il criterio individuale; SOL no. Il segnale dipende quindi anche
   dall'asset e non giustifica una conclusione universale su rialzi e ribassi.

Questo restringe le spiegazioni del fallimento precedente; non prova che tutta
la causa fosse la complessità o che nessun altro uso del flusso possa funzionare.
Il confronto ora misura direttamente il segno, distingue la fiducia dalla
classificazione e permette l'astensione nei casi deboli.

## Metodo fissato

- Input: rendimento dell'ultimo minuto diviso per RMS dei 60 rendimenti precedenti;
  numero di scambi del minuto diviso per media dei 60 minuti precedenti.
- Regimi: valore assoluto del rendimento normalizzato con soglie 0,5 / 1 / 2,
  attività con soglie 0,5 / 1,5. I movimenti negativi vengono riflessi per stimare
  le frequenze di inversione, invarianza e continuazione insieme a quelli positivi.
- Stima: conteggi con 100 pseudo-osservazioni verso le frequenze aggregate;
  prior globale add-one. Movimenti osservati nulli ricevono la frequenza
  incondizionata e non generano un segnale selezionato.
- Selezione: probabilità assoluta di rialzo o ribasso almeno 52%; altrimenti
  astensione. La selezione usa soltanto il modello stimato sul passato.
- Causalità: anchor ogni cinque minuti, ultima candela disponibile termina prima
  dell'anchor, rendimento misurato da anchor + 1 minuto ad anchor + 6 minuti.
  Solo etichette maturate entro il primo del mese entrano nel fit. Finestre
  complete; nessun riempimento dei buchi. Gli esiti non si sovrappongono nel
  singolo confronto.
- Criterio: copertura almeno 5%, accuratezza almeno 52%, limite inferiore 95%
  dell'accuratezza sopra 50% e del guadagno Brier sopra zero, log-loss non peggiore,
  vantaggio in almeno cinque mesi con almeno 100 chiamate ciascuno, almeno 100
  previsioni in entrambe le direzioni. Il trasferimento complessivo richiede il
  superamento da parte di **entrambi** gli asset aggiuntivi.
- Incertezza: bootstrap appaiato di blocchi mobili di sette giorni, 2.000 repliche,
  semi fissati nei protocolli. Gli intervalli sono condizionati ai modelli salvati:
  non includono l'incertezza dell'intero processo di addestramento né correggono
  tutte le esplorazioni storiche precedenti.

ETH e SOL sono correlati a BTC: non sono repliche indipendenti né dati prospettici.
Il rendimento usa prezzi di chiusura delle candele, non prezzi eseguibili. Non
sono stati simulati ordini, spread, commissioni o slippage. Un minuto di ritardo
nel target non dimostra una latenza reale end-to-end. Il risultato non prova
redditività; nessun ordine o servizio è stato attivato.

## Evidenze e riproduzione

- [Protocollo iniziale](micro/protocol.json), [risultati completi](micro/results.json).
- [Protocollo simmetrico](reversal/protocol.json), [risultati BTC](reversal/results.json).
- [Protocollo di trasferimento](transfer/protocol.json), [risultati ETH/SOL](transfer/results.json).
- [Diagnostica per direzione](direction-diagnostics.json), [verifica indipendente](verification.json).
- [Esempio storico ETH](preview-eth.json).

I protocolli sono stati registrati rispettivamente il 27 ottobre 2026 alle
08:58:31, 09:02:26 e 09:06:07 UTC. Gli hash legano codice, sorgenti, conteggi,
probabilità e risultati. I file NPZ contengono soltanto array numerici e non
richiedono pickle. Previsioni ed esiti sono in file distinti; la loro separazione
non trasforma questo backtest in una prova prospettica.

Verifica completa degli artefatti pubblicati, senza inferenza o download:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python \
  scripts/verify_direction_research.py
```

La verifica indipendente non importa lettori, modelli, feature o metriche di
produzione. Ricostruisce **11 confronti e 77 fit mensili**, inclusi campioni,
cutoff, conteggi, probabilità, selezioni, esiti, score, bootstrap e criteri.
Con `--raw` verifica anche i 334 Parquet BTC originali e i 22 ZIP nativi conservati
localmente. Quest'ultimo controllo è stato eseguito; richiede gli archivi locali
indicati dai protocolli, che non vengono duplicati nel repository.

Durante il controllo indipendente è emersa una differenza aritmetica del
verificatore su un rapporto di attività esattamente pari a 1,5. Sommare i conteggi
interi prima di dividere per 60 evita che una convoluzione con pesi arrotondati
sposti quel campione oltre la soglia. Il verificatore è stato corretto; codice
congelato, previsioni e risultati degli esperimenti sono rimasti invariati.

Esempio di previsione storica, senza leggere file degli esiti:

```bash
.venv-timestral/bin/python scripts/preview_direction_forecast.py \
  --symbol ETHUSDT --anchor 2026-07-31T23:50:00Z
```

Restituisce ribasso 47,04%, invarianza 0,37%, rialzo 52,59%, con segnale di rialzo
per il rendimento tra 23:51 e 23:56 UTC. Il fit è quello del 1° luglio. È un
singolo esempio tecnico, non una misura aggiuntiva di accuratezza. Il comando
rifiuta contesti incompleti, modelli futuri, anchor disallineati e target che
entrano nel periodo protetto.

Gli eseguibili `run_timestral_microdirection.py`, `run_timestral_reversal.py` e
`run_direction_transfer.py` conservano le fasi di registrazione, preparazione e
valutazione. Richiedono nuove directory di output e non sovrascrivono i risultati.
I primi due riutilizzano gli archivi/cache indicati nei protocolli; il terzo
acquisisce soltanto i 22 URL mensili fissati. Per una semplice revisione dei
risultati bastano il verificatore e gli array pubblicati.

## Dati, spazio e protezioni

I minuti ETH/SOL coprono settembre 2025–luglio 2026: 480.960 minuti per asset,
nessun buco o candela scartata, 42.542.011 byte di download complessivo. I checksum
ufficiali sono stati verificati. Formato e unità temporali seguono la
[documentazione Binance Public Data](https://github.com/binance/binance-public-data).
Sono candele OHLCV, non eventi di order book.

Il Brier multiclass usa la somma dei tre errori quadratici senza divisione per
due, coerentemente con la [definizione documentata da scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#brier-score-loss).
Un suo miglioramento non basta da solo a dimostrare calibrazione perfetta.

Spazio libero misurato prima/dopo l'acquisizione e la copia degli artefatti:
49,36 → 49,26 GiB. Nuovi output locali circa 88 MiB, report pubblicabile circa
46 MiB; misure fisiche e spazio libero possono differire per APFS e attività
concorrenti. Nessun modello o ambiente installato, nessun dato cancellato.

Nessuna sorgente o etichetta dal **1° agosto 2026** in avanti è stata letta.
I precedenti esperimenti e i loro hash restano invariati; preservati anche
`.env` e i due file locali dell'utente. Fra i file già presenti sono cambiati
soltanto README e workflow CI. Verifiche locali: **222 test superati, 8 test con
PostgreSQL saltati** perché il database locale non è disponibile. La CI include
i test con TimescaleDB reale e i tre verificatori delle evidenze salvate.

Resta da verificare la persistenza del segnale ETH su nuovi esiti registrati
dopo le previsioni e la sua eventuale utilità economica. Gli holdout protetti
non sono stati utilizzati per colmare questa lacuna.
