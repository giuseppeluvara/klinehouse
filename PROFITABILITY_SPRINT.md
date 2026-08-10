# HISTORICAL PROFITABILITY SPRINT — protocollo congelato

## Perimetro

Questo sprint confronta esclusivamente `PF_XBTUSD` con sei candidati storici:
Ridge, HistGradientBoosting e Transformer, ciascuno con holding massimo di una
o quattro ore. Non contiene paper trading, simulazione della coda, WebSocket,
API private o invio ordini. I risultati sono condizionati all'avvenuto fill
maker e non dimostrano fillability reale.

Gli esperimenti sequence e rapid evidence, i rispettivi dataset, bundle,
risultati, il commit `1a4cd9c723d94e5280c5532b5a56c24d5ade027b` e il tag
`sequence-t0-2026-08-08` sono input immutabili e non vengono letti dal modello.

## Data gate, prima del training

Il prodotto deve risultare dall'API pubblica Kraken Derivatives come
`PF_XBTUSD`, `flexible_futures`, base BTC, quote USD, `contractSize=1`,
`contractValueTradePrecision=4`, negoziabile, non scaduto e permesso sulle
piattaforme EEA `europa` o `europa_crypto` senza un divieto paese dichiarato.

Le sole fonti ammesse sono:

- candle trade e mark Kraken Charts a un minuto;
- analytics Kraken trade-count e CVD a un minuto;
- ZIP ufficiale Kraken “Export historical funding rates”;
- analytics funding Kraken, usato anche per verificare la sovrapposizione.

Ogni risposta viene paginata, gli estremi inclusivi vengono deduplicati e i
timestamp sono normalizzati in UTC con semantica `[start, end)`. Non sono
ammessi forward-fill, interpolazione, provider alternativi o sostituzione di
feature. Si conservano URL, conteggi raw/deduplicati e SHA-256.

Il periodo modellabile parte dal primo trimestre UTC completo comune alle
fonti e termina all'inizio dell'ultimo trimestre UTC completo. Ogni fonte deve
coprire almeno il 99,5% dei minuti di ciascun trimestre e almeno il 95% degli
anchor teorici deve restare valido in ogni fold. Funding deve essere presente
per ogni holding interval e l'export deve sovrapporsi all'API con timestamp e
relative rate concordanti entro `1e-12`. Servono almeno 24 mesi iniziali e
cinque trimestri completi di validation. Un solo fallimento produce
`data_no_go` e vieta dataset building e training.

## Dataset e split

Gli anchor sono orari. Ogni input contiene i 120 minuti completi che precedono
l'anchor e sette canali causali: log-return trade, log-return mark, range trade
normalizzato, `log1p(volume)`, `log1p(trade_count)`, imbalance aggressore
`(buy-sell)/(buy+sell)` e basis close trade/mark. Qualunque minuto o valore
richiesto mancante invalida l'anchor.

Ridge e HGB ricevono ultimo valore, media, deviazione standard population e
pendenza OLS per ciascun canale, esattamente 28 feature. Il Transformer riceve
la sequenza `120×7`.

Lo split è expanding walk-forward: i primi 24 mesi completi sono training e
ogni trimestre UTC completo successivo è un fold. Il trimestre corrente
incompleto è soltanto diagnostico. Per tutti i candidati si applica il purge
più severo: `train_label_end <= validation_start - 120 minuti`, usando un
label-end di quattro ore. Ogni validation label deve terminare entro il fold.
Non esistono label censurate a destra.

## Modelli congelati

- Ridge: `alpha=1`, scaler train-only.
- HGB: 200 iterazioni, 31 foglie, learning rate 0,05, L2 1, seed 7.
- Transformer: `d_model=32`, quattro teste, due layer, feed-forward 64,
  dropout 0,1, mean pooling, SmoothL1, AdamW `3e-4`, weight decay `1e-2`,
  batch 256, dieci epoche, clipping 1, seed 7/17/29 e media delle predizioni.
- Target continuo: rendimento lineare futuro espresso in bps, a 1h o 4h.
- Policy: long-only quando lo score supera il percentile train-only 97,5,
  stimato sugli ultimi due trimestri completi del training; una posizione alla
  volta.

Non sono consentiti tuning, nuove feature, nuovi modelli, nuove soglie o costi
dopo l'apertura dei risultati.

## Backtest e costi

La quantità è
`floor((100 / entry_price) / 0.0001) * 0.0001 BTC`; il notional iniziale non
supera 100 USD. Il PnL lordo è lineare:
`quantity * (exit_price - entry_price)`.

L'ingresso assume un fill maker al trade-open dell'anchor. Senza stop, l'uscita
assume un fill maker al trade-close dell'ultimo minuto dell'orizzonte. Lo stop
long a -1% è attivato dal primo mark-low che attraversa la soglia; l'uscita
taker usa il peggiore tra stop price e trade-open del minuto successivo. Un
prezzo mancante invalida il trade.

Costi in USD:

- uscita ordinaria: 2 bps maker in ingresso e 2 bps maker in uscita;
- stop: 2 bps maker in ingresso e 5 bps taker in uscita;
- adverse-selection reserve: 4 bps del notional iniziale per ogni trade;
- funding: il maggiore tra la somma dei soli relative rate long positivi
  realmente maturati e 1 bp per ora effettiva di holding, applicato al
  notional iniziale. I crediti sono ignorati.

## Inferenza statistica e gate

Si usa bootstrap cronologico settimanale entro fold, 10.000 repliche, seed 7.
Sono riportati il CI descrittivo 95% e il CI family-wise a 99,1667%, con
quantili 0,4167% e 99,5833%.

Un candidato passa soltanto con almeno 100 trade, almeno 10 trade in almeno
cinque fold, PnL netto positivo in almeno cinque fold e negli ultimi due,
limite inferiore del CI 99,1667% sopra zero e profit factor almeno 1,1.

Tra più candidati promossi vince il limite inferiore del CI 99,1667% più alto.
Entro 1 bp si preferiscono Ridge, HGB, Transformer e poi 1h, 4h. Nessun
candidato promosso produce `historical_no_go`; un vincitore viene rifittato sui
trimestri completi e congelato, senza costruire infrastruttura paper/live.

