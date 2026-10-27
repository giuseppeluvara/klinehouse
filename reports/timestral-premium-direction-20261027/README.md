# Nuova informazione dai derivati: premium index e direzione spot

Ho acquisito una fonte nuova per questi esperimenti, il **premium index dei
futures perpetui Binance**, e verificato se migliora la distinzione rialzo/ribasso
su BTC, ETH e SOL nei periodi gennaio–luglio 2024, 2025 e 2026.

**Il metodo provato non fornisce un segnale aggiuntivo stabile.** Nessuno dei
nove confronti supera il criterio registrato di informazione aggiuntiva rispetto
al solo spot. Il guadagno medio di Brier condizionale è negativo in tutti e nove,
con intervalli che includono lo zero: questo non prova l'assenza assoluta di
informazione nella fonte, ma non sostiene il suo impiego con questo metodo.

Anche la valutazione completa, che include i prezzi invariati, peggiora: quattro
criteri direzionali superati su nove, contro sei del controllo spot.
Gli holdout dal **1° agosto 2026 in avanti restano intatti**.

## Che cosa è nuovo e come viene isolato

La [documentazione Binance](https://developers.binance.info/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)
descrive le candele del premium index con timestamp di apertura e chiusura e
valori OHLC. I campi successivi non rappresentano volumi di negoziazione:
questo esperimento non li usa come volume o flusso di ordini. I file mensili e
i checksum provengono dall'[archivio pubblico ufficiale](https://github.com/binance/binance-public-data).

Al modello vengono aggiunte dodici feature prefissate: livello del premium,
medie su 5/30/120 minuti, variazioni su 5/30 minuti, scostamento dalla media,
contesto degli altri due asset e interazioni con attività e rendimento spot.
Il premium è espresso in punti base e limitato a ±10 prima degli aggregati.

Il confronto usa due modelli logistici binari, con gli stessi iperparametri:

1. **Spot:** le ventitré feature già disponibili.
2. **Spot + premium:** le stesse feature più le dodici nuove.

Entrambi sono ristimati ogni mese sui novanta giorni precedenti, con esiti già
maturati, pooling dei tre asset e simmetria rialzo/ribasso. Normalizzazione e
parametri sono stimati soltanto sui campioni di addestramento. Non ci sono
identificativi dell'asset o sweep di configurazioni. Sono stati eseguiti
**42 fit: 3 anni × 7 mesi × 2 modelli**.

Il modello binario impara solo dagli esiti con movimento, per stimare
`P(rialzo | prezzo non invariato)`. Queste probabilità sono poi ricombinate con
la **stessa probabilità di prezzo invariato**, presa dal modello comune congelato,
per entrambi i concorrenti. Tutti gli anchor e tutti i prezzi invariati rimangono
nella valutazione multiclass. Il guadagno non può quindi derivare da una diversa
stima della classe invariata.

Il target resta il rendimento spot di cinque minuti, da un minuto a sei minuti
dopo l'anchor. Il nuovo input ammette soltanto candele premium chiuse da almeno
un minuto, quindi con apertura non successiva a `anchor − 120 secondi`.
**Latenza storica effettiva e revisioni della fonte non sono verificabili da
questi archivi:** il replay è causale rispetto a questa assunzione dichiarata,
non prova una disponibilità operativa misurata in tempo reale.

## Tutti i nove confronti

La colonna condizionale riguarda tutti gli esiti con movimento, senza selezione
in base alla sicurezza; non indica una copertura utilizzabile conoscendo in
anticipo quali prezzi resteranno invariati. Le altre colonne includono i prezzi
invariati e applicano la soglia originale del 52%.

| Periodo / asset | Accuratezza rialzo/ribasso, condizionale | Accuratezza completa sui casi selezionati | Copertura | Criterio completo spot | Criterio completo con premium |
|---|---:|---:|---:|---|---|
| 2024 BTC | 51,87% | 52,71% | 30,95% | Superato | Non superato |
| 2024 ETH | 51,35% | 52,38% | 29,98% | Non superato | Non superato |
| 2024 SOL | 51,11% | 51,20% | 23,07% | Non superato | Non superato |
| 2025 BTC | 50,80% | 51,59% | 30,88% | Non superato | Non superato |
| 2025 ETH | 51,65% | 53,70% | 28,53% | Superato | Superato |
| 2025 SOL | 51,10% | 51,58% | 24,58% | Superato | Non superato |
| 2026 BTC | 51,70% | 52,95% | 46,16% | Superato | Superato |
| 2026 ETH | 52,52% | 53,70% | 44,37% | Superato | Superato |
| 2026 SOL | 52,34% | 52,61% | 19,95% | Superato | Superato |

Accuratezza superiore al 50% non significa che la fonte nuova contribuisca.
Il confronto appaiato con il solo spot sui medesimi esiti condizionali dà:

| Periodo / asset | Guadagno medio Brier | Intervallo 99,444% del guadagno Brier |
|---|---:|---:|
| 2024 BTC | −0,0000664 | [−0,0001404; +0,0000134] |
| 2024 ETH | −0,0000417 | [−0,0001257; +0,0000616] |
| 2024 SOL | −0,0000856 | [−0,0001908; +0,0000255] |
| 2025 BTC | −0,0000694 | [−0,0001601; +0,0000173] |
| 2025 ETH | −0,0000452 | [−0,0001332; +0,0000398] |
| 2025 SOL | −0,0000376 | [−0,0001334; +0,0000488] |
| 2026 BTC | −0,0000760 | [−0,0001668; +0,0000156] |
| 2026 ETH | −0,0000731 | [−0,0001567; +0,0000090] |
| 2026 SOL | −0,0000234 | [−0,0001358; +0,0000802] |

Sono differenze assolute di Brier binario: positivo significa migliore.
Anche tutti e nove gli intervalli del guadagno di log-loss rispetto al controllo
spot includono lo zero. I risultati dettagliati, compresi mesi, confronto con
frequenze storiche e criteri completi, sono in [results.json](gap-safe/results.json).

## Protocollo, dati mancanti e conservazione dei campioni

Prima del fitting è stato registrato un criterio aggiuntivo, distinto da quello
multiclass precedente: tutti e nove i confronti devono avere limite inferiore
dell'accuratezza condizionale sopra il 50%, guadagni Brier e log-loss con limite
inferiore positivo rispetto a spot e frequenze storiche, almeno cinque mesi con
guadagno Brier positivo e almeno cento previsioni per entrambe le direzioni.
Il criterio completo precedente deve inoltre passare su tutti e nove per
sostenere una soluzione generale; non è stato sostituito dal nuovo criterio.

Gli intervalli condizionali usano 4.000 repliche appaiate a blocchi mobili di
sette giorni, seed 271032, correzione Bonferroni sui nove asset/anni per ciascun
endpoint. Non correggono l'intera storia esplorativa né tutti gli endpoint
congiuntamente e non includono rifitting durante il bootstrap.

La prima preparazione si è fermata sui quattordici minuti mancanti nel file BTC
di novembre 2023. Codice e [protocollo iniziali](protocol.json) sono conservati,
con [ricevuta del fallimento](failure.json). **Prima di qualsiasi fitting o
metrica** è stato registrato il [protocollo con gestione dei buchi](gap-safe/protocol.json):

- il contesto deve contenere 120 minuti consecutivi disponibili su tutti i tre
  asset; il premium non viene propagato o interpolato;
- entrambi i modelli si addestrano sugli stessi campioni con contesto valido;
- quando il contesto manca in valutazione, il concorrente con premium usa la
  previsione del controllo spot;
- tutti gli anchor originali vengono conservati e verificati per identità.

I 90 ZIP coprono ottobre–luglio per ciascun periodo e asset. Contengono
**3.939.798 righe**, con **4.362 minuti mancanti** rispetto alle griglie attese:
14 per asset nel novembre 2023 e 1.440 per asset nel giugno 2026.
Non sono stati rimossi esiti di valutazione. Il ricorso al solo spot riguarda
**312 anchor per asset nel 2026**, circa lo 0,51%; nessuno nel 2024 o 2025.
Le origini, i checksum e i conteggi sono in [sources.json](gap-safe/sources.json).

## Verifica e riproduzione

Il verificatore è indipendente dai modelli e dai reader di produzione. Usa una
griglia con valori mancanti espliciti e convoluzioni per ricostruire il contesto,
contro la selezione su timestamp e le somme cumulative del runner. Controlla:

- checksum ufficiali e contenuto dei 90 ZIP, unità temporali e OHLC;
- timestamp e identità degli esiti rispetto agli esperimenti congelati;
- campioni di addestramento uguali, label mature, normalizzazione e gradienti;
- 42 fit convessi, previsioni e fallback, metriche, intervalli e criteri;
- conservazione del tentativo interrotto e confine protetto.

Il gradiente massimo verificato è circa `1,03e-7`; non viene rieseguita
l'ottimizzazione. La [ricevuta](gap-safe/verification.json) conferma l'audit anche
dei file originali locali. In CI si verificano i dati e gli output pubblicati
senza scaricare nuovamente gli ZIP.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python scripts/verify_premium_direction.py
# Aggiungere --raw per controllare anche i 90 ZIP originali locali.
```

Per rifare preparazione, fitting e valutazione in una destinazione nuova:

```bash
.venv-timestral/bin/python scripts/run_premium_direction_gaps.py register --root research/out/premium-replay
.venv-timestral/bin/python scripts/run_premium_direction_gaps.py prepare --root research/out/premium-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python scripts/run_premium_direction_gaps.py run --root research/out/premium-replay
```

Il replay rifiuta sovrascritture e riusa gli ZIP locali verificandoli con i
checksum ufficiali. L'audit ordinario controlla il bundle pubblicato completo,
incluso il tentativo iniziale conservato; il replay non ricrea quella cronologia.

Suite locale: **260 test superati, 8 test PostgreSQL saltati** per assenza del
database locale. La CI include il database e gli audit degli studi precedenti.
Il fitting e la valutazione completa richiedono **39,09 secondi**, con massimo
RSS misurato di circa **1,30 GiB** su questa macchina; non è una misura della
latenza di produzione.

## Spazio e limiti

Payload ZIP circa 74,1 MiB; bundle pubblicato circa 41,2 MiB, per circa 115,3 MiB
prima degli oggetti Git. Nessuna installazione o download di modelli.
Il controllo dello spazio libero passa da 40,69 a 39,58 GiB: la differenza è
maggiore dei file prodotti e non è attribuibile interamente a questa ricerca
con le misure disponibili. Nessun file o processo estraneo è stato eliminato.

Le prove sono retrospettive su anni e prezzi già esplorati. Il risultato riguarda
questa rappresentazione del premium e questa classe di modelli; non esclude
ogni possibile uso dei derivati. Non sono state misurate redditività, esecuzione
o performance prospettiche. **Il candidato non viene promosso come soluzione
generale:** la fonte nuova, nella forma provata, non stabilizza il segnale
rialzo/ribasso tra periodi.
