# Adattamento delle previsioni e test storico separato

**Selezione congelata; test gennaio–marzo 2026 ancora da completare.**

Il fit iniziale usa ottobre–novembre 2025 e la selezione usa soltanto dicembre. Tra nove candidati preregistrati è stato scelto un modello per orizzonte, poi rifittato sui soli dati ottobre–dicembre. La cartella dei dati di test era assente alla registrazione del protocollo: `2026-10-25T18:25:33.905534Z`.

| Orizzonte | Candidato congelato | MAE dicembre | MAE prezzo invariato |
|---|---|---:|---:|
| 4h | Timestral prezzo × 0,5 + correzione bias | 61,5928 bps | 61,6614 bps |
| 24h | Timestral con covariate originale | 137,6896 bps | 138,8029 bps |

Sono risultati di selezione, non evidenza del test. I coefficienti, gli scaler e le distribuzioni di calibrazione non verranno aggiornati con gli esiti del test.

- [Protocollo, comandi e limiti](../../docs/timestral-adaptation.md)
- [Protocollo congelato](frozen/protocol.json)
- [Parametri e selezione completa](frozen/bundle.json)
- [Hash pubblicati prima del test](freeze-sha256.json)

Il disegno è motivato dall'analisi già osservata di maggio–luglio 2026. Gennaio–marzo è un periodo storico separato per questa selezione, non un test prospettico; non si può escludere contaminazione nel pretraining di Timestral. Il vecchio holdout agosto 2026–febbraio 2027 resta protetto.
