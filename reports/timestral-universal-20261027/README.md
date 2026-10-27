# Un modello comune: risultato positivo nel 2026, generalizzazione incompleta

Ho implementato un modello con **gli stessi coefficienti, input e soglia di
selezione per BTC, ETH e SOL**, ristimato ogni mese sul passato. Supera il criterio
su tutti e tre nel gennaio–luglio 2026. La replica congelata negli anni precedenti
fallisce: **non è ancora una soluzione generale verificata**.

Il controllo aggiuntivo conta: fermarsi al solo 2026 avrebbe prodotto una
conclusione troppo forte. Sono conservati anche tutti i risultati negativi.
Gli holdout dal 1° agosto 2026 in avanti restano intatti.

## Risultati della regola comune

Target: rialzo, prezzo invariato o ribasso nei cinque minuti successivi a un
minuto di ritardo. La previsione si astiene quando nessuna probabilità direzionale
raggiunge il 52%. I prezzi invariati rimangono negli esiti e penalizzano una
chiamata direzionale sbagliata.

| Asset | Accuratezza 2024 | Criterio 2024 | Accuratezza 2025 | Criterio 2025 | Accuratezza 2026 | Criterio 2026 |
|---|---:|---|---:|---|---:|---|
| BTC | 53,23% | Superato | 51,90% | Non superato | 52,91% | Superato |
| ETH | 52,65% | Non superato | 54,02% | Superato | 53,87% | Superato |
| SOL | 51,59% | Non superato | 52,60% | Superato | 52,99% | Superato |

Ogni coorte considera gli anchor di gennaio–luglio dell'anno indicato. La tabella
misura l'accuratezza **sui casi selezionati**, non su tutti gli anchor.
L'accuratezza da sola non determina il superamento: occorrono anche probabilità
utili, copertura minima e stabilità mensile. Il modello passa **6 controlli su 9**;
la replica temporale congelata passa **3 dei 6 controlli aggiuntivi**. Il criterio
generale richiede tutti i controlli, non la media dei risultati.

Il risultato principale del 2026, prima della replica:

| Asset | Chiamate / anchor | Copertura | Accuratezza | Intervallo 98,333% | Guadagno Brier: intervallo 98,333% |
|---|---:|---:|---:|---:|---:|
| BTC | 27.035 / 61.055 | 44,28% | 52,91% | 52,30–53,59% | +0,132–+0,544% |
| ETH | 25.595 / 61.055 | 41,92% | 53,87% | 53,27–54,49% | +0,476–+0,894% |
| SOL | 11.113 / 61.055 | 18,20% | 52,99% | 52,05–54,11% | +0,310–+1,147% |

Il guadagno Brier confronta la probabilità prevista con le frequenze di classe
stimate sul passato, sugli stessi campioni selezionati. Anche il confronto con
il precedente modello simmetrico migliora su tutti e tre nel 2026: limiti
inferiori del guadagno rispettivamente +0,095%, +0,309%, +0,220%.

## Che cosa è cambiato e che cosa spiegano le prove

Il vecchio modello a dodici celle usava essenzialmente il movimento dell'ultimo
minuto e l'attività. Questo usa rendimenti normalizzati su 1, 5, 15 e 60 minuti,
flusso taker su 1 e 5 minuti, attività, frequenza recente di prezzi invariati e
alcune interazioni. Il modello principale aggiunge il contesto degli altri due
mercati. Non riceve l'identità dell'asset.

Prima dell'esecuzione ho registrato quattro confronti:

1. `local`: stessi tredici input propri, modello separato per asset;
2. `shared`: tredici input propri, un unico modello per i tre asset;
3. `cross`: ventitré input con contesto degli altri mercati, unico modello;
   questo è il confronto principale;
4. `leave_one_out`: esclude dall'addestramento tutte le etichette dell'asset da
   prevedere, mantenendo gli input di mercato disponibili nel passato.

Nel 2026 passano tutti i modelli `local`, `shared` e `cross`. Perciò non posso
attribuire il miglioramento soltanto all'addestramento congiunto o ai dati degli
altri asset. Il contesto e la rappresentazione continua meritano attenzione;
l'effetto causale delle singole feature non è isolato da questo esperimento.

Il test senza etichette dell'asset da prevedere passa su BTC ed ETH, ma fallisce
su SOL: accuratezza 51,61%, log-loss 0,8714 contro 0,8229 della baseline. Una
diagnostica descrittiva mostra che questo modello assegna a SOL una probabilità
media di prezzo invariato dello 0,53%, contro il 3,86% osservato; usando anche
lo storico SOL la stima è 4,15%. Una regola comune può quindi richiedere dati
rappresentativi di ogni mercato: trasferirla a un asset mai incluso non è già
verificato.

La replica temporale aggiunge un limite diverso. Su SOL 2024 la probabilità
media della direzione scelta è 53,15%, ma l'accuratezza è 51,59%; su BTC 2025
53,16% contro 51,90%. Sono evidenze descrittive di fiducia eccessiva in quei
periodi, non la prova di un'unica causa di tutti i fallimenti.

## Variante non lineare, senza ritoccare le soglie

Dopo il fallimento della replica lineare ho registrato una sola variante:
gradient boosting multiclass condiviso, 120 iterazioni, learning rate 0,03,
massimo 7 foglie, almeno 200 campioni per foglia, regolarizzazione L2 10,
63 bin, nessun early stopping, seme fissato. Usa gli stessi input e criteri;
la media delle previsioni speculari impone la simmetria fra rialzi e ribassi.

La variante passa **5 dei 9 controlli**, con accuratezze selezionate:

| Asset | 2024 | 2025 | 2026 |
|---|---:|---:|---:|
| BTC | 53,25% — passa | 52,26% — fallisce | 53,03% — passa |
| ETH | 52,47% — fallisce | 53,80% — passa | 53,95% — passa |
| SOL | 51,62% — fallisce | 51,78% — fallisce | 52,01% — passa |

La maggiore complessità non risolve quindi la generalizzazione. Questa variante
è un'esplorazione dichiarata successiva ai risultati lineari, non una nuova
conferma indipendente sugli stessi periodi.

## Protocollo e criteri mantenuti

- Fit mensili sugli ultimi 90 giorni; etichette maturate entro il cutoff.
- Candele chiuse prima dell'anchor; ritorno da anchor + 1 a anchor + 6 minuti.
  Gli esiti non si sovrappongono all'interno del singolo confronto.
- Stima lineare: regressione logistica multinomiale, C=0,1, LBFGS, massimo
  500 iterazioni, tolleranza 1e-8; standardizzazione soltanto sul train e clipping
  a ±8. Augmentation speculare dei soli input direzionali e delle etichette.
- Identiche regole sui tre asset: copertura almeno 5%, accuratezza almeno 52%,
  limiti inferiori degli intervalli sopra 50% per accuratezza e zero per guadagno
  Brier, log-loss non peggiore della frequenza, almeno cento chiamate in ciascuna
  direzione e vantaggio in almeno cinque mesi con cento chiamate ciascuno.
- Bootstrap appaiato a blocchi mobili di sette giorni, 4.000 repliche, stessi
  giorni campionati sui tre asset. Intervalli 98,333% con correzione Bonferroni
  per i tre asset nella singola coorte. Non correggono tutte le esplorazioni
  precedenti e sono condizionati ai modelli già stimati.
- Nessuna variante viene dichiarata generale perché passa soltanto nel 2026.
  Nessuna soglia è stata abbassata dopo i risultati.

Il protocollo e gli input congelati sono in [protocol.json](protocol.json);
la replica è in [history/protocol.json](history/protocol.json) e la variante
in [trees/protocol.json](trees/protocol.json). I risultati completi sono in
[results.json](results.json), [history/results.json](history/results.json) e
[trees/results.json](trees/results.json).

## Verifica e uso tecnico

I verificatori indipendenti ricostruiscono feature, campioni, target, cutoff,
membership del train, coefficienti e probabilità, selezioni, metriche e bootstrap.
Per i modelli lineari controllano anche la stazionarietà dell'obiettivo convesso;
per gli alberi percorrono gli split numerici esportati senza pickle e senza
riaddestrare gli alberi. Nessun verificatore importa il lettore, il modello o
le metriche dell'esperimento.

Sono verificati **91 fit dei candidati**, **42 fit di baseline aggiuntivi** e
**27 confronti asset/famiglia/coorte**. Il controllo locale `--raw` ha anche
ricostruito i 66 ZIP nuovi dopo verifica dei checksum ufficiali. Gli esiti sono
in [verification.json](verification.json) e
[history-verification.json](history-verification.json).

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python \
  scripts/verify_universal_direction.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python \
  scripts/verify_universal_history.py --trees
```

Il secondo comando accetta `--raw` quando gli archivi originali locali indicati
dalla provenienza sono presenti. I due comandi ordinari verificano i dati e
gli output pubblicati senza nuovi download o fitting.

Il motore comune può essere esercitato su una data storica con:

```bash
.venv-timestral/bin/python scripts/preview_shared_direction.py \
  --anchor 2026-07-31T23:50:00Z
```

Calcola le feature soltanto dalle 63 candele chiuse necessarie per ciascun asset;
non apre file degli esiti. Le sorgenti compresse sono storiche e vengono caricate
e controllate per hash prima di estrarre il contesto. Restituisce tre previsioni
usando **lo stesso fit `cross_7_all`**, probabilità e astensione. Convalida cutoff,
allineamento, integrità e confine protetto. L'[esempio salvato](preview.json)
è una verifica tecnica del percorso di previsione, non ulteriore evidenza di
accuratezza né una previsione emessa prima dell'esito.

## Dati, limiti e stato

Riutilizzati i minuti settembre 2025–luglio 2026 già verificati. Aggiunti
66 ZIP nativi di BTC/ETH/SOL, ottobre 2023–luglio 2025: **132.192.052 byte**
scaricati, **964.800 minuti per asset**, nessun buco o candela scartata.
Millisecondi prima del 2025 e microsecondi dal 2025 sono gestiti esplicitamente,
secondo la [documentazione ufficiale Binance](https://github.com/binance/binance-public-data).
Nessuna copia CSV estratta e nessuna nuova installazione di modelli o ambienti.

Nuovi file locali dell'esperimento circa 212 MiB, report circa 84 MiB, esclusi
i nuovi oggetti Git. La stima iniziale era inferiore a 0,5 GiB. Il controllo
dello spazio libero è passato da 47,28 a 39,52 GiB; un controllo successivo ha
misurato circa 27 GiB. Sono presenti circa 5,22 GiB di swap, senza misura iniziale
comparabile. La variazione non può essere attribuita ai soli file prodotti da
questa ricerca; nessun dato o processo estraneo è stato modificato e non sono
necessari altri download per verificare gli artefatti.

Verifiche locali: **240 test superati, 8 test PostgreSQL saltati** per assenza del
database locale. La CI esegue anche gli otto test con TimescaleDB reale e tutti
i verificatori degli esperimenti precedenti e di questa ricerca. Il confronto
degli hash di 392 file preesistenti conferma che sono cambiati soltanto README
e workflow CI; `.env`, file dell'utente e precedenti esperimenti sono preservati.

Gli anni 2024 e 2025 avevano già fornito prezzi orari agli studi precedenti:
la replica su minuti acquisiti dopo il congelamento del metodo resta una prova
retrospettiva. Gli asset sono correlati. Il test riguarda un orizzonte di cinque
minuti e questi tre mercati; non dimostra capacità su qualunque strumento o
orizzonte. Un vantaggio di direzione non dimostra redditività dopo i costi.
Nessuna esecuzione, commissione, spread o slippage è stata simulata.

L'implementazione e gli esperimenti sono completati e verificati; **l'obiettivo
di una soluzione predittiva generale resta aperto**. I risultati recenti sono
utilizzabili come candidati di ricerca, con l'astensione e i limiti dichiarati.
Le prove prospettiche e la validazione economica non sono state eseguite.
Gli holdout protetti non sono stati letti per ottenere una conferma favorevole.
