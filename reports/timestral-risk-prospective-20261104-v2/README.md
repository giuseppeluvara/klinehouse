# Klinehouse: prova paper settimanale Timestral

La prova è stata chiusa il 9 novembre 2026 dopo la prima previsione mancata. Il registro si è fermato prima di qualsiasi ingresso paper. La descrizione sotto documenta l’attivazione e il calendario originario. [Stato alla chiusura](../klinehouse-closeout-20261109.json)

Il processo locale paper è stato avviato e il suo handle è stato verificato vivo. Il registro contiene l’attivazione e la risposta originale del controllo dell’orologio Binance: **zero previsioni prospettiche, zero ingressi paper, zero esiti maturi**. Il profitto storico non viene contato come profitto futuro.

Il capitale simulato è **10.000 USDT per ciascun portafoglio alternativo**, senza leva. Timestral è confrontato con EWMA e pesi fissi a parità di trend. I tre portafogli sono scenari alternativi, non un investimento complessivo di 30.000 USDT.

| Evento | Data e ora italiana |
|---|---|
| Prima previsione e salvataggio | 9 novembre 2026, 01:00 |
| Primo ingresso simulato, se la previsione è valida | 10 novembre 2026, 01:00 |
| Prima maturazione settimanale | 17 novembre 2026, 01:00 |
| Liquidazione paper finale | 5 gennaio 2027, 01:00 |
| Scadenza automatica del processo | 5 gennaio 2027, 01:10 |

Regole e pesi del modello sono congelati. Le nuove decisioni possono usare soltanto candele già chiuse e calibrazioni con esito già maturo. La previsione deve essere salvata e riconosciuta entro dieci minuti dall’anchor del lunedì, prima dell’ingresso del martedì. L’ingresso paper usa esclusivamente quotazioni ricevute dopo l’inizio della finestra, entro sessanta secondi. Gli slot persi restano mancanti; nessuna previsione o esecuzione viene ricostruita retroattivamente.

Il calcolo paper consuma il bid/ask visualizzato, aggiunge commissione di 10 bp e riserva avversa di 2 bp, e verifica che il portafoglio si autofinanzi. Usa quantità frazionarie, non ordini conformi ai lotti di borsa: non attesta fill reali né eseguibilità simultanea sui due asset. Il controllo di rischio osserva giornalmente il NAV di liquidazione. Alla perdita osservata del 25% dal massimo, simula la liquidazione ai prezzi disponibili e ferma quel portafoglio. Gap, ritardi e osservazioni mancanti possono portare oltre il 25%; non è un limite garantito.

Verifiche completate:

- Sette test, incluso il ciclo previsione → riconoscimento del salvataggio → ingresso → esito, il rifiuto dei dati futuri e il superamento della soglia per un gap avverso.
- Il nuovo motore riproduce 377 decisioni storiche: differenza massima dei pesi 3,56e-15.
- Due risposte sorgenti e 52 osservazioni di calibrazione verificate; prima/intermedia/ultima inferenza riprodotte con errore zero.
- Registro append-only, hash del codice/dati/pesi e lettura coerente del journal mentre il processo è in esecuzione. La catena locale non esclude la riscrittura dell’intero database senza una ricevuta esterna.
- Prova con profondità reale disponibile oggi: lo scenario Timestral da 10.000 USDT comportava circa 5,60 USDT di costi di ingresso ipotetici. Era un controllo fuori calendario; nessun portafoglio è stato modificato e non conta come operazione.

La calibrazione storica comprende prezzi pubblici agosto–novembre 2026, esplicitamente ammessi solo in questa nuova prova. Gli artefatti dei vecchi esperimenti protetti rimangono chiusi, ma quelle date non possono essere definite non osservate per l’intero programma di ricerca. Il primo bundle è stato conservato e sostituito prima dell’attivazione per correggere lock e gestione dei ritardi; nessun parametro finanziario è stato selezionato dopo esiti futuri.

Il processo è locale, separato dal precedente shadow pilot, che non è stato riavviato. Non installa servizi di sistema, cron o automazioni dell’app. Un arresto del computer o del processo può interromperlo: i buchi rimarranno visibili. La sua disponibilità futura non è garantita dalla sola configurazione.

Il replay storico precedente rimane positivo, ma fallisce il gate statistico e non dimostra un vantaggio di Timestral su EWMA. Il protocollo prevedeva otto settimane di osservazione. La prova si è chiusa prima di produrre dati di portafoglio prospettici.

Fonti tecniche: [Timestral ufficiale](https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff), [Binance market data](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market).
