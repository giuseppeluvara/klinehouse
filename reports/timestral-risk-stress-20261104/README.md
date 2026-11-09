# Timestral: stress dei costi e delle date di ingresso

Questa analisi conserva gli stessi segnali, pesi e date del replay precedente. Non cambia il processo paper in esecuzione e non costituisce una nuova strategia o una validazione fuori campione.

| Moltiplicatore commissioni e slippage | NAV finale USDT | CAGR netto trading | Drawdown con mark giornalieri e agli ingressi/uscite |
|---|---:|---:|---:|
| 0× | 22439.53 | 13.11% | 17.90% |
| 1× | 22054.32 | 12.82% | 18.13% |
| 2× | 21675.74 | 12.52% | 18.36% |
| 5× | 20578.60 | 11.63% | 19.04% |
| 10× | 18872.23 | 10.17% | 20.17% |

Base: commissioni 10 bp e slippage avverso 2 bp per lato. Il caso 10× ipotizza 100 bp e 20 bp per lato. Lo zero è soltanto un riferimento senza attriti, non un costo eseguibile. In questo replay il rischio di regime e la durata dell’investimento sono più rilevanti dell’ipotesi di costi base.

Ogni finestra sotto riparte da 10.000 USDT liquidi e applica i pesi storici già congelati, addebitando ingresso e liquidazione finale. Non viene selezionata la data migliore.

| Orizzonte | Finestre sovrapposte | Con saldo netto positivo | Peggior rendimento netto | Miglior rendimento netto | Finestre sopra CAGR 5% |
|---|---:|---:|---:|---:|---:|
| 52 settimane | 291 | 195 | -13.88% | 80.57% | 148 |
| 156 settimane | 187 | 187 | 11.62% | 79.98% | 182 |

Tutte le finestre Timestral restano entro il 25% sui mark giornalieri del replay. Il risultato non garantisce un limite intragiornaliero o futuro. Le finestre si sovrappongono e condividono gran parte dei prezzi: 187/187 non significa probabilità di successo del 100% e non sostituisce un intervallo statistico basato su campioni indipendenti.

Il contributo del 2020 è significativo: eliminando soltanto il suo fattore moltiplicativo dal percorso storico, l’incremento residuo è circa il 49,24%, annualizzato al 7,45% per gennaio 2021–luglio 2026. È attribuzione descrittiva del percorso esistente, non un nuovo portafoglio che riparte liquido nel 2021.

I rendimenti sui veri confini del calendario differiscono dai precedenti gruppi di settimane assegnate all’anno di ingresso. Con i mark di fine anno, Timestral registra circa −7,55% nel 2022 e −2,10% nel 2025. I valori −7,90% e −4,99% del rapporto originale appartengono ai gruppi settimanali del replay e restano conservati con quella definizione; non sono stati riscritti nei risultati originali.

Confronti preservati: EWMA ha 201/291 finestre annuali positive contro 195/291 di Timestral; entrambi hanno 187/187 finestre triennali positive nello storico esaminato. Il vantaggio aggiuntivo di Timestral rimane non dimostrato.

Verifiche: otto hash di risposte sorgenti controllati; 342 decisioni congelate riutilizzate; tutte e quattro le simulazioni con costi base riproducono i NAV originali entro 1e-6 USDT. Gli altri scenari sono diagnostiche di sensibilità, non fill osservati. I risultati complessivi, compresi confronti e finestre sfavorevoli, sono salvati nei JSON.

La prova prospettica resta invariata: prima previsione 9 novembre 2026 alle 01:00 italiane, primo ingresso paper 10 novembre, primo esito settimanale 17 novembre. Il processo è stato ripollato vivo in questa continuazione; non è stato riavviato.

La discussione [Reddit r/algotrading sui costi](https://www.reddit.com/r/algotrading/comments/1vyeq3l/cost_model_in_backtesting_results/) e quella sulle [esecuzioni parziali e lo slippage](https://www.reddit.com/r/algotrading/comments/1vxnf8a/how_are_you_guys_accounting_for/) sono spunti metodologici, non prove dei risultati qui riportati né della nazionalità degli autori. Le misure del rapporto derivano dal replay locale verificato.

Obiettivo ancora attivo: storico positivo e sensibilità ai costi favorevole; nessun nuovo profitto prospettico disponibile.
