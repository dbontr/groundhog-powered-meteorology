# Groundhog Ensemble Forecast (GitHub Pages)

A static GitHub Pages site that:
- pulls **Groundhog Day predictions** (shadow = “longer winter”, no shadow = “early spring”) from the **GROUNDHOG-DAY.com API**
- scores each animal against an **outcome definition** (default: U.S. Feb+Mar warmth vs the 20th‑century average)
- learns **weights** from historical performance
- runs a **year-by-year backtest** (no peeking!) and computes a **Data Adequacy Epoch (DAE)**: the first year when there’s enough track record to start reporting “Since YEAR, we’ve been X% accurate”.

## Quick start

1) Create a new GitHub repo and copy this project into it.

2) Install Node (v18+), then run:

```bash
npm install
npm run update:predictions
# optional: npm run update:outcomes:us
npm run build
```

3) Commit + push.

4) Enable GitHub Pages:
- **Settings → Pages**
- Source: **Deploy from a branch**
- Branch: `main`
- Folder: `/docs`

Then open your Pages URL.

## What you get

- `/docs` is the site (pure HTML/CSS/JS).
- `/scripts` are data update helpers.
- A scheduled workflow (`.github/workflows/update-data.yml`) can refresh cached predictions automatically.

## Data sources

### Predictions
We use the (free, no-auth) endpoints documented at:

- `https://groundhog-day.com/api/v1/groundhogs`
- `https://groundhog-day.com/api/v1/groundhogs/{slug}`
- `https://groundhog-day.com/api/v1/predictions?year={year}`

See `scripts/update_predictions.mjs`.

### Outcomes (“what actually happened?”)
There isn’t one universally accepted definition of “early spring” (welcome to climatology: it’s messy).
The default outcome definition in this repo is:

> Compute a 6‑week temperature signal using **February + March** contiguous U.S. temperatures (relative to a baseline),
> and classify **early spring** if the *mean anomaly* is > 0, else **longer winter**.

This mirrors NOAA/NCEI’s *idea* of comparing Phil’s forecast to U.S. national temperatures, but it is still a choice you should be explicit about on the site.

- You can provide outcomes manually in `docs/data/outcomes.csv`, **or**
- Use `scripts/update_outcomes_us_conus.mjs` as a starting point (it pulls NOAA “Climate at a Glance” CSV-style time series URLs commonly used in tutorials and coursework). If the NOAA endpoint format changes, you’ll only need to tweak that script.

## Backtest & weighting

The backtest is strict:
- For year *t*, weights are trained on years `< t`.
- Only groundhogs with predictions that year participate.

Weighting methods (selectable in the UI):
- **Bayesian mean** (Beta prior) + sample-size boost (default, robust for new groundhogs)
- **Smoothed accuracy**
- **Exponentially-decayed accuracy** (recent years matter more)

## Data Adequacy Epoch (DAE)

DAE is computed as the *first year* when all of the following hold:
- At least `minGroundhogs` have at least `minObsPerGroundhog` historical scored predictions
- The ensemble’s rolling Wilson 95% CI half-width is below `maxCIHalfWidth`
- At least `minBacktestYears` backtest years exist

You can change the thresholds in the UI.

## License
MIT
