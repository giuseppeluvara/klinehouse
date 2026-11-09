# Timestral: test economico delle previsioni salvate

Nessuna delle sei regole supera il criterio registrato. Si tratta di dati storici già esplorati, non di un test prospettico.

Modello: autentiche previsioni Timestral-5M, codice e pesi fissati nel precedente esperimento capacity. BTCUSDT gennaio–luglio 2026; 1.271 anchor a 4 ore e 211 giornalieri, escludendo i due esiti sul confine di agosto. Entrata spot a +1 ora dall’anchor; uscita alla fine dell’orizzonte originale. Commissione 10 bp e slippage avverso 2 bp per lato. Ogni finestra include una pausa di un’ora: queste regole comportano quindi frequenti round trip. Non sono stime dell’esecuzione ottimale possibile.

| Regola | Orizzonte | Operazioni | Rendimento netto trading | Buy-and-hold continuo | Gate |
|---|---:|---:|---:|---:|---|
| median_positive | 4 h | 620 | -82.43% | -28.46% | Non superato |
| lower_quartile_above_cost | 4 h | 0 | 0.00% | -28.46% | Non superato |
| median_positive_vol_target | 4 h | 620 | -36.99% | -28.46% | Non superato |
| median_positive | 24 h | 94 | -41.16% | -26.40% | Non superato |
| lower_quartile_above_cost | 24 h | 0 | 0.00% | -26.40% | Non superato |
| median_positive_vol_target | 24 h | 94 | -10.69% | -26.40% | Non superato |

Il costo opportunità del capitale al 5% annuo è riportato separatamente nel JSON, non è un addebito osservato. Il filtro prudente non opera: zero operazioni non dimostra redditività. La riduzione della perdita tramite dimensionamento non prova un vantaggio rispetto a una semplice esposizione inferiore.

Verifiche: tre test unitari del calcolo economico, Mypy strict sul modulo, sei hash dei dati di prezzo e 4.446 righe di trade ricostruite con Decimal. Errore massimo della ricostruzione: 6,01e-16. Il bootstrap conserva blocchi settimanali, ma rimane descrittivo per questa ricerca post hoc.

Il passo successivo registrato è un esperimento distinto di allocazione settimanale BTC/ETH: trend fissato a 126 giorni, Timestral usato per prevedere il rischio, confronto a parità di trend con EWMA e pesi fissi. Verranno mantenuti anche eventuali risultati sfavorevoli.

Riferimenti:
- [Timestral: repository ufficiale](https://github.com/mistralai/timestral)
- [Mistral: annuncio del modello](https://mistral.ai/news/timestral/)
- [Fin-RSI: backtest validation](https://github.com/howard-lynn-ye/Fin-RSI/blob/master/fin_skills/_skills/backtest-validation/SKILL.md)
- [Discussione Reddit sui costi di trading](https://www.reddit.com/r/algotrading/comments/1vyeq3l/cost_model_in_backtesting_results/)
