# Groundhog Powered Meteorology

A static GitHub Pages forecast that combines Groundhog Day predictions with an explicit climate baseline.

The site:
- downloads animal predictions from the GROUNDHOG-DAY.com API
- scores them against NOAA contiguous-U.S. February–March temperature anomalies
- evaluates every model decision with chronological walk-forward testing
- starts from a 15-year rolling climatology probability
- gives the groundhog layer weight only after it improves prior-year Brier score
- ranks animals by skill relative to climatology during the same active years

This design prevents a large collection of correlated animals from appearing more predictive than a simple recent-climate baseline.

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

### 1. Recent-climate prior

For forecast year `t`, the model estimates the early-spring probability from only the previous 15 outcome years. A Beta(1,1) prior prevents probabilities of exactly zero or one.

### 2. Groundhog reliability

Each animal receives an empirical reliability signal from predictions before `t`. The model combines Bayesian accuracy, recency-weighted accuracy, a rolling window, stability, and sample-size evidence.

Below-chance animals can receive negative weights, which correctly treats them as contrarian signals. Animals without enough evidence are shrunk or excluded.

### 3. Climate guard

The groundhog layer is compared with climatology using prior-year Brier score. Its contribution is zero unless it has demonstrated incremental probabilistic skill. Evidence shrinkage prevents a short lucky streak from receiving a large weight.

### 4. Nested walk-forward selection

For each historical test year:
1. every feature uses outcomes strictly before that year
2. the tuning profile is selected using only earlier walk-forward predictions
3. the selected profile predicts the held-out year

Changing a future outcome therefore cannot change an earlier prediction. Tests enforce this property.

## Reported metrics

The site reports:
- ordinary accuracy
- balanced accuracy, so a dominant class cannot hide failure on the minority class
- Brier score for probability quality
- the matching 15-year climatology baseline
- current groundhog contribution

The leaderboard shows raw accuracy and climate-relative skill. Skill uses a smoothed accuracy estimate minus the climatology accuracy over the animal's active years.

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
- `docs/data/` — cached predictions, animal directory, and outcomes
- `scripts/` — data updates, evaluator, and build helper
- `tests/` — Node test suite

The Python evaluator is only a compatibility wrapper around `scripts/eval_fusion.mjs`, preventing the model from diverging across two implementations.

## License

MIT
