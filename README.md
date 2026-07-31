# Groundhog Powered Meteorology

A static GitHub Pages forecast powered entirely by Groundhog Day predictions, with climatology retained only as an evaluation baseline.

The site:
- downloads animal predictions from the GROUNDHOG-DAY.com API
- scores them against NOAA contiguous-U.S. February–March temperature anomalies
- evaluates every prediction with chronological walk-forward testing
- combines an 80% reliability-weighted animal vote with a 20% all-animal crowd vote
- uses no weather or climatology input when producing the forecast
- ranks animals by skill relative to climatology during the same active years

The forecast is therefore genuinely groundhog powered while the displayed climate baseline keeps its performance in context.

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

### 1. Reliability-weighted vote

For forecast year `t`, each animal is scored only on predictions made before `t`. The primary vote uses a Wilson-confidence reliability weight with:

- a four-year decay half-life
- a 20-year maximum history window
- at least eight prior observations
- sample-size shrinkage
- signed weights, so consistently below-chance animals become contrarian signals

This reliability vote supplies 80% of the final probability.

### 2. Full crowd vote

The remaining 20% comes from the unweighted share of all animals predicting early spring. This keeps the forecast tied to the complete Groundhog Day field and makes the model less brittle when individual historical weights are noisy.

When too few animals have usable histories, the model automatically falls back toward the crowd vote. Both inputs are animal predictions; climatology never enters the forecast calculation.

### 3. Walk-forward evaluation

For every historical test year, all reliability statistics are trained using years strictly before that year. The held-out year is then predicted once. Changing a future outcome therefore cannot alter an earlier prediction, and tests enforce that property.

## Reported metrics

The site reports:
- ordinary accuracy
- balanced accuracy, so a dominant class cannot hide failure on the minority class
- Brier score for probability quality
- the matching 15-year climatology baseline
- reliability-vote and crowd-vote composition

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
