# Klinehouse: audit indipendente e supervisione futura

Il 9 novembre 2026 il processo e la supervisione sono stati disattivati. Questo documento conserva le verifiche di preparazione e il calendario previsto in origine. [Stato alla chiusura](../klinehouse-closeout-20261109.json)

Il processo della prova Timestral è stato ripollato vivo e non è stato riavviato. Il runtime e i parametri finanziari congelati rimangono identici.

Il nuovo verificatore ricostruisce indipendentemente con Decimal cassa, quantità, commissioni, consumo dei livelli bid/ask, pesi effettivi e arresti per rischio. Non importa il motore di esecuzione del portafoglio. Sei test comprendono transazioni su più livelli del book, un gap avverso e la liquidazione finale; verificano anche il rifiuto di saldi/fee alterati, pesi strategici sbagliati, asset scambiati, profondità insufficiente e arresti di rischio non riportati. Insieme ai sette test del pilot, 13 test mirati superati.

L'audit del registro reale è coerente, ma contiene ancora zero ingressi paper, zero fill paper e zero esiti maturi: non è una certificazione di profitto futuro. Ciascuno dei tre portafogli alternativi conserva 10.000 USDT simulati e quantità zero. La versione precedente del verificatore è conservata separatamente.

È stata creata e verificata nell'app la supervisione **Supervisione Timestral Klinehouse**, in questa stessa chat, ogni giorno alle **00:30 ora italiana**, fino al controllo conclusivo del pilot. Il prompt richiede verifiche del processo vivo e audit in sola lettura, impedisce writer duplicati e ricostruzioni retroattive, e mantiene il silenzio se lo stato non cambia. Segnalerà nuovi guasti, anomalie, decisioni, ingressi, esiti o interventi necessari.

La supervisione non sostituisce il processo locale che gestisce gli orari di mercato. I task locali richiedono il computer acceso e l'app in esecuzione, come indicato nella [documentazione ufficiale OpenAI sulle attività pianificate](https://learn.chatgpt.com/docs/automations?surface=app). L'automazione non autorizza ordini, accesso ad account o uso di capitale reale.

Il calendario prevedeva la prima previsione il 9 novembre 2026 alle 01:00, il primo ingresso paper il 10 novembre alle 01:00 e il primo esito settimanale il 17 novembre alle 01:00, ora italiana. La prima previsione è saltata e la prova è terminata senza ingressi paper o esiti maturi.
