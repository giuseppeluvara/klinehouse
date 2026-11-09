# Timestral: allocazione BTC/ETH con rischio e costi espliciti

Il replay storico è positivo: 10.000 USDT diventano 22.054,32 USDT. La redditività prospettica e il vantaggio aggiuntivo di Timestral restano non dimostrati. Il candidato non supera il gate statistico registrato.

Vincoli confermati dall’utente: capitale 10.000 USDT, niente leva, drawdown massimo 25%.

Periodo eseguito: 7 gennaio 2020–28 luglio 2026, 342 settimane. Universo BTCUSDT/ETHUSDT. Ogni lunedì si osservano esclusivamente le chiusure già disponibili; si eseguono i nuovi pesi all’apertura del martedì. Trend positivo a 126 giorni; altrimenti peso zero. Obiettivo di volatilità 20% annuo, peso massimo 50% per asset; il residuo resta liquido e non produce interessi.

Timestral-5M prevede congiuntamente il logaritmo dei rendimenti giornalieri al quadrato dei due asset: contesto 256 giorni, otto giorni previsti. Si utilizzano i giorni 2–8 per il rischio della finestra effettivamente detenuta. La calibrazione usa soltanto le ultime 52 previsioni settimanali con esito già maturo, con un minimo di 26. Non è un modello Gamma o una regressione logistica rinominata Timestral.

Le commissioni ipotizzate sono 10 bp per fill, più 2 bp di slippage avverso. Vengono applicate solo alle quantità cambiate. Il portafoglio è autofinanziato: commissioni e slippage riducono il capitale prima di fissare le quantità, senza prestiti impliciti. Le posizioni vengono interamente liquidate alla fine.

| Metodo | NAV finale USDT | CAGR netto trading | Drawdown chiusure giornaliere | Limite superiore conservativo da high/low giornalieri |
|---|---:|---:|---:|---:|
| Timestral + trend | 22054.32 | 12.82% | 18.13% | 18.94% |
| EWMA + stesso trend | 22668.16 | 13.29% | 18.31% | 19.26% |
| Pesi fissi + stesso trend | 119267.90 | 45.93% | 51.70% | 53.41% |
| Buy-and-hold iniziale 50/50 | 106478.80 | 43.42% | 77.77% | 80.34% |

Il limite high/low ipotizza estremi simultanei sui due asset e l’ordine intragiornaliero peggiore. È un limite conservativo per il portafoglio ipotetico ricostruito, non un drawdown realmente osservato durante l’esecuzione. Timestral ed EWMA rispettano il 25% nel replay; pesi fissi e buy-and-hold lo superano.

Il CAGR netto trading di Timestral è 12,82%, contro un costo opportunità fissato al 5% annuo. La differenza è 7,82 punti percentuali annui: non equivale al rendimento di un finanziamento reale né incorpora imposte, costo dei dati o infrastruttura. I costi di esecuzione addebitati nel replay Timestral sommano 307,55 USDT.

La cautela decisiva è statistica: il CI 99% della media del rendimento settimanale Timestral è circa [-0,0543%; +0,6031%], quindi include zero. Il CI 99% della differenza di utilità rispetto a EWMA è [-0,0462%; +0,0343%] a settimana e include anch’esso zero. EWMA guadagna di più a punto stimato. Non attribuiamo a Timestral un vantaggio dimostrato e non selezioniamo un nuovo parametro dopo questi risultati.

Gruppi di settimane attribuite all’anno di ingresso negativi: 2022 −7,90%, 2025 −4,99%. Non sono rendimenti sugli esatti confini di calendario: la successiva analisi con mark al 31 dicembre misura rispettivamente −7,55% e −2,10%. I risultati numerici originali restano conservati. Nel tratto 2026 di questo replay il filtro trend rimane liquido: rendimento di trading zero, non rendimento garantito. La scelta dello storico non implica che quei prezzi fossero assenti dal pretraining del modello.

Verifica indipendente: otto hash delle risposte sorgenti, 404 righe di previsione e relative finestre di calibrazione, 1.368 righe di portafoglio. Il NAV è stato ricostruito con una soluzione algebrica per le regioni buy/sell, indipendente dal solver a bisezione del motore. Tre inferenze autentiche del modello, prima/intermedia/ultima, sono state riprodotte con errore massimo zero. Tre test unitari del motore di portafoglio e Mypy strict superati.

Il primo tentativo si era fermato prima di ogni inferenza per una candela BTCUSDT parziale dell’8 febbraio 2018. Il tentativo fallito è conservato. Questa versione esclude senza imputazione i contesti e gli esiti che toccano barre parziali. Anche la precedente prova Timestral ad alta frequenza, fallita su tutte e sei le regole, resta conservata.

Fonti e metodo:

- [Timestral, codice ufficiale](https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff)
- [Pesi Timestral-5M fissati](https://huggingface.co/mistralai/Timestral-5M/tree/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f)
- [Binance, dati pubblici spot](https://developers.binance.com/en/docs/binance-spot-api-docs/rest-api/market-data-endpoints)
- [Fin-RSI, backtest-validation](https://github.com/howard-lynn-ye/Fin-RSI/blob/master/fin_skills/_skills/backtest-validation/SKILL.md)
- [Fin-RSI, execution-cost-analysis](https://github.com/howard-lynn-ye/Fin-RSI/blob/master/fin_skills/_skills/execution-cost-analysis/SKILL.md)
- [Reddit: discussione sui costi](https://www.reddit.com/r/algotrading/comments/1vyeq3l/cost_model_in_backtesting_results/)

Stato obiettivo: attivo. Abbiamo un replay profittevole con Timestral e con il rischio storico sotto il limite richiesto; mancano evidenza prospettica e un contributo incrementale stabile di Timestral. Nessun ordine è stato inviato.

Contesto accademico consultato: [Time series momentum, Moskowitz–Ooi–Pedersen (2012)](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf) e [Volatility-Managed Portfolios, Moreira–Muir (2017)](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.12513). Questo esperimento non replica esattamente quei lavori e i loro risultati non validano i rendimenti crypto qui misurati.
