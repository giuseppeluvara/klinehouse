# Klinehouse: audit carry del 4 novembre 2026

Stato: redditività della strategia non dimostrata. Nessun ordine eseguito.

Analisi storica su tre finestre consecutive di 30 giorni, dal 6 agosto al 4 novembre 2026 UTC. Universo fissato BTC, ETH, SOL; quantità spot e short perpetual abbinate. Ogni finestra parte da 10.000 USDT di spot e 20.000 USDT di capitale allocato. I prezzi sono proxy candle-open, non fill verificati.

| Asset | Finestra UTC | Funding USDT | Basis USDT | Netto di esecuzione | Dopo costo capitale |
|---|---|---:|---:|---:|---:|
| BTCUSDT | 2026-08-06 | 70.70 | 2.06 | 30.36 | -51.83 |
| BTCUSDT | 2026-09-05 | 42.10 | 0.33 | 2.80 | -79.39 |
| BTCUSDT | 2026-10-05 | 41.28 | -4.04 | 1.24 | -80.96 |
| ETHUSDT | 2026-08-06 | 57.59 | 1.78 | 15.93 | -66.26 |
| ETHUSDT | 2026-09-05 | 40.81 | -0.33 | 0.39 | -81.80 |
| ETHUSDT | 2026-10-05 | 23.40 | -1.10 | -12.54 | -94.73 |
| SOLUSDT | 2026-08-06 | 36.77 | 5.40 | -2.97 | -85.16 |
| SOLUSDT | 2026-09-05 | 32.27 | -2.94 | -12.33 | -94.52 |
| SOLUSDT | 2026-10-05 | 34.18 | -2.47 | -1.79 | -83.98 |

Commissioni ipotizzate per fill: spot 10 bp, perpetual 5 bp; slippage 2 bp per fill su entrambe le gambe. Costo annuo del capitale 5% su 20.000 USDT: è un costo opportunità ipotizzato, non un addebito osservato. Cinque finestre hanno saldo positivo dopo esecuzione ma prima del costo opportunità. Tutte e nove risultano negative dopo tale costo. Questo distingue il saldo simulato dalla convenienza economica.

Correzioni: nuovo accounting con Decimal e validazione dei tipi/valori; funding sulla quantità × mark di ogni settlement; basis e quattro fill contabilizzati; annualizzazione scanner basata sui tempi osservati e sull’intervallo configurato. Il vecchio carry book rimane una semplice illustrazione del funding, ora etichettata come tale.

Verifiche: sei test unitari superati, compilazione Python, 31 hash delle risposte API verificati e nove identità di PnL ricalcolate. Mypy 2.4.1 in modalità strict supera il controllo del nuovo modulo carry_accounting.py. L’intera suite del repository non è stata eseguita.

Limiti: nessuna simulazione intraperiodo di margine/liquidazione; funding completeness non certificata; nessun order book o fill reale; assenza di holdout intatto per queste nuove ipotesi. I risultati non validano né escludono ogni possibile strategia carry.

Il progetto include anche un pilot shadow interrotto il 30 ottobre per spazio disco. In questa sessione non è stato riavviato. Gli artefatti dei vecchi holdout non sono stati aperti; tuttavia le quotazioni pubbliche e i funding agosto–novembre usati nel nuovo audit si sovrappongono alle date di vecchi protocolli protetti. Quel periodo non può più essere descritto come completamente non osservato per l’intero programma di ricerca. La copia locale non contiene .git e research/out non è presente: non equivale al precedente runtime completo.

Risorse consultate:

- [Binance: funding history API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History)
- [Reddit r/algotrading: cost model](https://www.reddit.com/r/algotrading/comments/1vyeq3l/cost_model_in_backtesting_results/): spunti sui costi, non prova di rendimento né verifica della nazionalità degli autori.
- [Fin-RSI backtest-validation](https://github.com/howard-lynn-ye/Fin-RSI/blob/master/fin_skills/_skills/backtest-validation/SKILL.md)
- [Fin-RSI backtest-overfitting](https://github.com/howard-lynn-ye/Fin-RSI/blob/master/fin_skills/_skills/backtest-overfitting/SKILL.md)

L’utente ha chiarito che “typesafe” significava Timestral. Il nuovo esperimento Timestral è riportato separatamente; i controlli di tipo dell’accounting rimangono utili ma non sostituiscono il modello richiesto.

Prossimo lavoro: registrare le ipotesi prima dei nuovi test, verificare completezza funding e margine intraperiodo, quindi validare i candidati su dati successivi alla selezione. Nessun candidato viene dichiarato profittevole sulla base dello scanner.
