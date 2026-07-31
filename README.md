# Groundhog Powered Meteorology

A static GitHub Pages forecast powered only by verified living groundhogs, with climatology retained strictly as an evaluation baseline.

The site:
- keeps only API entries marked `isGroundhog: 1`
- rejects null and missing prediction records instead of treating them as early-spring votes
- scores valid groundhog predictions against NOAA contiguous-U.S. February–March temperature anomalies
- evaluates every model decision with chronological walk-forward testing
- weights groundhogs by Wilson-confidence reliability with recent years emphasized
- uses no weather or climatology input when producing the forecast
- ranks groundhogs by skill relative to climatology during the same active years

The forecast is literally groundhog derived; ducks, lobsters, mascots, statues, and other non-groundhog forecasters are excluded.

## Quick start

Requires Node.js 18 or newer.

```bash
npm run update:predictions
npm run update:outcomes:us
npm run check
```

Useful commands:

```bash
npm test                  # model invariants and leakage tests
npm run evaluate          # current backtest and forecast metrics
npm run build             # seed missing data files
npm run check             # test, evaluate, and build
```

Enable GitHub Pages from the `main` branch and `/docs` folder.

## Forecast design

### 1. Verified-groundhog ingestion

The updater retains only API records where `isGroundhog` equals `1`. A prediction is accepted only when the API explicitly supplies shadow or no-shadow; null records such as “No Record” are discarded.

### 2. Reliability-weighted vote

For forecast year `t`, each groundhog is scored only on predictions made before `t`. The forecast uses a Wilson-confidence reliability weight with:

- a three-year decay half-life
- a 20-year maximum history window
- at least eight prior observations
- stronger sample-size evidence weighting
- signed weights, so consistently below-chance groundhogs become contrarian signals

The published probability comes entirely from this verified-groundhog reliability vote. If no groundhog has enough history, the fallback is the current verified-groundhog majority vote.

### 3. Walk-forward evaluation

For every historical test year, all reliability statistics are trained using years strictly before that year. The held-out year is then predicted once. Changing a future outcome therefore cannot alter an earlier prediction, and tests enforce that property.

## Reported metrics

The site reports:
- ordinary accuracy
- balanced accuracy, so a dominant class cannot hide failure on the minority class
- Brier score for probability quality
- the matching 15-year climatology baseline
- the number of reporting groundhogs with usable reliability histories

On the current cleaned data, the 2000–2025 walk-forward result is 88.5% accuracy and 93.8% balanced accuracy. The leaderboard shows raw accuracy and climate-relative skill over each groundhog's active years.

## Performance

Filtering non-groundhogs and missing records reduced the prediction cache from about 430 KB to 211 KB, or from 17.0 KB to 9.1 KB when gzipped. The complete backtest and current forecast average under 1 ms on Jupiter, and browser requests now use normal HTTP caching.

## Data sources

Predictions:
- `https://groundhog-day.com/api/v1/groundhogs`
- `https://groundhog-day.com/api/v1/predictions?year={year}`

Outcomes come from NOAA Climate at a Glance national monthly temperature anomalies.

The current binary target is:

> `EARLY_SPRING` when the mean of February and March CONUS anomalies is above 0°F relative to the 1901–2000 baseline; otherwise `LONG_WINTER`.

This target is transparent but imperfect. It covers roughly two months, includes February 1 before the ceremony, uses a national rather than local outcome, and is affected by long-term warming. A future version should use daily, location-aware observations over the exact 42 days after Groundhog Day.

## Repository layout

- `docs/` — static site and browser model
- `docs/lib/fusion.js` — canonical forecast and evaluation implementation
- `docs/data/` — verified-groundhog predictions, groundhog directory, and outcomes
- `scripts/` — data updates, evaluator, and build helper
- `tests/` — Node test suite

The Python evaluator is only a compatibility wrapper around `scripts/eval_fusion.mjs`, preventing the model from diverging across two implementations.

## License

MIT
