import { parseCSV } from "./lib/stats.js";
import { indexOutcomes, indexPredictions, runBacktest, trainWeights, predictionToOutcome } from "./lib/backtest.js";

const $ = (id) => document.getElementById(id);

const TARGET = "US_CONUS_FEBMAR_MEAN_ANOM";
const DEFAULT_OPTS = {
  minObs: 5,
  betaPrior: [2, 2],
  halfLifeYears: 10,
  alpha: 2.0,
  gamma: 0.5
};

const ALGORITHMS = [
  { id: "bayes", label: "Bayesian mean + sample boost" },
  { id: "smooth_acc", label: "Smoothed accuracy" },
  { id: "exp_decay", label: "Exponentially-decayed accuracy" }
];

async function loadJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} (${url})`);
  return await res.json();
}

async function loadText(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} (${url})`);
  return await res.text();
}

function fmtPct(x, digits = 1) {
  if (!Number.isFinite(x)) return "—";
  return `${(100 * x).toFixed(digits)}%`;
}

function fmtPctValue(x, digits = 1) {
  if (!Number.isFinite(x)) return "—";
  const useDigits = Math.abs(x - Math.round(x)) < 1e-6 ? 0 : digits;
  return `${x.toFixed(useDigits)}%`;
}

function setStatus(msg) {
  const el = $("status");
  if (el) el.textContent = msg;
}

function outcomeLabel(outcome) {
  return outcome === "EARLY_SPRING" ? "EARLY SPRING" : "LATE WINTER";
}

function evaluateAlgorithms(predByYear, outcomes) {
  return ALGORITHMS.map((algo, index) => {
    const rows = runBacktest(predByYear, outcomes, TARGET, algo.id, DEFAULT_OPTS);
    let last = null;
    for (let i = rows.length - 1; i >= 0; i--) {
      if (Number.isFinite(rows[i].cumAcc)) {
        last = rows[i];
        break;
      }
    }
    if (!last && rows.length) last = rows[rows.length - 1];
    return {
      algo,
      rows,
      last,
      accuracy: last?.cumAcc ?? Number.NaN,
      backtestN: last?.cumN ?? 0,
      lastYear: last?.year ?? null,
      order: index
    };
  });
}

function pickBestAlgorithm(results) {
  const viable = results.filter((r) => Number.isFinite(r.accuracy));
  if (!viable.length) return null;
  viable.sort((a, b) => {
    if (b.accuracy !== a.accuracy) return b.accuracy - a.accuracy;
    if (b.backtestN !== a.backtestN) return b.backtestN - a.backtestN;
    return a.order - b.order;
  });
  return viable[0];
}

function computeNowcast(predByYear, outcomes, algoResult) {
  const years = Array.from(predByYear.keys());
  if (!years.length) return null;
  const latestYear = Math.max(...years);
  const { weights } = trainWeights(predByYear, outcomes, TARGET, latestYear, algoResult.algo.id, DEFAULT_OPTS);

  const preds = predByYear.get(latestYear) ?? [];
  let earlyWeight = 0;
  let totalWeight = 0;
  let used = 0;
  let usedWeighted = false;

  for (const p of preds) {
    const w = weights.get(p.groundhogSlug);
    if (!w) continue;
    used++;
    totalWeight += w;
    usedWeighted = true;
    const predOut = predictionToOutcome(!!p.shadow);
    if (predOut === "EARLY_SPRING") earlyWeight += w;
  }

  if (!totalWeight && preds.length) {
    // Fallback: simple majority vote when weights are unavailable.
    used = preds.length;
    totalWeight = preds.length;
    earlyWeight = preds.reduce((acc, p) => {
      const predOut = predictionToOutcome(!!p.shadow);
      return acc + (predOut === "EARLY_SPRING" ? 1 : 0);
    }, 0);
  }

  const pEarly = totalWeight ? earlyWeight / totalWeight : 0.5;
  const pred = pEarly >= 0.5 ? "EARLY_SPRING" : "LONG_WINTER";
  const certainty = totalWeight ? Math.max(pEarly, 1 - pEarly) : Number.NaN;

  return { latestYear, pred, certainty, used, totalWeight, usedWeighted };
}

function computeLeaderboard(predByYear, outcomes, groundhogDir) {
  const nameBySlug = new Map();
  for (const g of (groundhogDir?.groundhogs ?? [])) {
    if (g?.slug) nameBySlug.set(g.slug, g.name || g.slug);
  }
  for (const preds of predByYear.values()) {
    for (const p of preds) {
      if (!p?.groundhogSlug) continue;
      if (!nameBySlug.has(p.groundhogSlug) && p.groundhogName) {
        nameBySlug.set(p.groundhogSlug, p.groundhogName);
      }
    }
  }

  const stats = new Map();
  for (const [year, preds] of predByYear) {
    const actual = outcomes.get(`${TARGET}:${year}`);
    if (!actual) continue;
    for (const p of preds) {
      const slug = p.groundhogSlug;
      if (!slug) continue;
      if (!stats.has(slug)) stats.set(slug, { n: 0, k: 0 });
      const s = stats.get(slug);
      s.n += 1;
      const predOut = predictionToOutcome(!!p.shadow);
      if (predOut === actual) s.k += 1;
    }
  }

  const rows = Array.from(stats.entries()).map(([slug, s]) => ({
    slug,
    name: nameBySlug.get(slug) ?? slug,
    n: s.n,
    k: s.k,
    accuracy: s.n ? s.k / s.n : Number.NaN
  }));

  rows.sort((a, b) => {
    if (b.accuracy !== a.accuracy) return b.accuracy - a.accuracy;
    if (b.n !== a.n) return b.n - a.n;
    return a.name.localeCompare(b.name);
  });

  return rows;
}

function renderLeaderboard(rows) {
  const table = $("leaderboard");
  if (!table) return;

  if (!rows.length) {
    table.innerHTML = `
      <thead>
        <tr>
          <th class="rank">Rank</th>
          <th>Groundhog</th>
          <th>Accuracy</th>
          <th>Obs</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td colspan="4">No scored groundhog predictions yet.</td>
        </tr>
      </tbody>
    `;
    return;
  }

  const body = rows.map((g, idx) => {
    const accuracy = Number.isFinite(g.accuracy) ? g.accuracy * 100 : Number.NaN;
    return `
      <tr>
        <td class="rank">${String(idx + 1).padStart(2, "0")}</td>
        <td>${g.name}</td>
        <td class="accuracy">${fmtPctValue(accuracy, 1)}</td>
        <td>${g.n}</td>
      </tr>
    `;
  }).join("");

  table.innerHTML = `
    <thead>
      <tr>
        <th class="rank">Rank</th>
        <th>Groundhog</th>
        <th>Accuracy</th>
        <th>Obs</th>
      </tr>
    </thead>
    <tbody>
      ${body}
    </tbody>
  `;
}

async function run() {
  try {
    setStatus("Loading data...");

    const [predObj, outcomesText, groundhogDir] = await Promise.all([
      loadJson("./data/predictions.json"),
      loadText("./data/outcomes.csv"),
      loadJson("./data/groundhogs.json")
    ]);

    const outcomesRows = parseCSV(outcomesText);
    const outcomes = indexOutcomes(outcomesRows);
    const predByYear = indexPredictions(predObj);

    const leaderboardRows = computeLeaderboard(predByYear, outcomes, groundhogDir);
    renderLeaderboard(leaderboardRows);

    const scoredYears = Array.from(predByYear.keys()).filter((y) => outcomes.has(`${TARGET}:${y}`));
    const minYear = scoredYears.length ? Math.min(...scoredYears) : null;
    const maxYear = scoredYears.length ? Math.max(...scoredYears) : null;
    const leaderboardMeta = scoredYears.length
      ? `Computed from groundhog-day.com predictions vs NOAA CAG outcomes. Scored years: ${minYear}–${maxYear}.`
      : "Computed from groundhog-day.com predictions vs NOAA CAG outcomes.";
    $("leaderboardMeta").textContent = leaderboardMeta;

    const isSample = String(predObj.updatedAt || "").includes("SAMPLE");
    if (isSample) {
      setStatus("Sample data loaded — run npm run update:predictions for the full leaderboard.");
    }

    const results = evaluateAlgorithms(predByYear, outcomes);
    let best = pickBestAlgorithm(results);
    let usedFallback = false;

    if (!best) {
      usedFallback = true;
      const algo = ALGORITHMS[0];
      best = {
        algo,
        rows: [],
        last: null,
        accuracy: Number.NaN,
        backtestN: 0,
        lastYear: null,
        order: 0
      };
    }

    const nowcast = computeNowcast(predByYear, outcomes, best);
    if (!nowcast) {
      setStatus("No prediction data available.");
      return;
    }

    const indicator = $("indicator");
    indicator.textContent = outcomeLabel(nowcast.pred);
    document.body.dataset.outcome = nowcast.pred;

    $("certainty").textContent = fmtPct(nowcast.certainty, 1);
    $("algoAccuracy").textContent = fmtPct(best.accuracy);
    $("callYear").textContent = `Forecast for ${nowcast.latestYear}`;
    $("algoName").textContent = best.algo.label;
    $("algoAcc").textContent = Number.isFinite(best.accuracy)
      ? `Backtest accuracy: ${fmtPct(best.accuracy)} (n=${best.backtestN}, through ${best.lastYear})`
      : "Backtest unavailable (outcomes missing).";
    $("predictionYear").textContent = `${nowcast.latestYear}`;
    $("voterCount").textContent = `${nowcast.used}`;
    $("meta").textContent = usedFallback
      ? "Backtest unavailable — showing default algorithm."
      : (nowcast.usedWeighted
        ? "Prediction uses the highest-accuracy algorithm from the backtest set."
        : "Backtest available, but weights missing — using simple majority vote.");

    if (!isSample) setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(String(err));
  }
}

run();
