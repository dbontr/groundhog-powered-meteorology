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

const NOAA_FALLBACK = {
  source: "NOAA Heritage: Grading Groundhogs",
  period: "2005-2024",
  groundhogs: [
    { name: "Staten Island Chuck", accuracy: 85.0 },
    { name: "General Beauregard Lee", accuracy: 80.0 },
    { name: "Lander Lil", accuracy: 75.0 },
    { name: "Concord Charlie", accuracy: 65.0 },
    { name: "Gertie the Groundhog", accuracy: 65.0 },
    { name: "Jimmy the Groundhog", accuracy: 60.0 },
    { name: "Woodstock Willie", accuracy: 60.0 },
    { name: "Buckeye Chuck", accuracy: 55.0 },
    { name: "French Creek Freddie", accuracy: 55.0 },
    { name: "Malverne Mel", accuracy: 55.0 },
    { name: "Octoraro Orphie", accuracy: 52.63 },
    { name: "Dunkirk Dave", accuracy: 50.0 },
    { name: "Holtsville Hal", accuracy: 50.0 },
    { name: "Poor Richard", accuracy: 50.0 },
    { name: "Uni the Groundhog", accuracy: 47.37 },
    { name: "Schnogadahl Sammi", accuracy: 38.89 },
    { name: "Punxsutawney Phil", accuracy: 35.0 },
    { name: "Woody the Woodchuck", accuracy: 35.0 },
    { name: "Mojave Max", accuracy: 25.0 }
  ]
};

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

  for (const p of preds) {
    const w = weights.get(p.groundhogSlug);
    if (!w) continue;
    used++;
    totalWeight += w;
    const predOut = predictionToOutcome(!!p.shadow);
    if (predOut === "EARLY_SPRING") earlyWeight += w;
  }

  const pEarly = totalWeight ? earlyWeight / totalWeight : 0.5;
  const pred = pEarly >= 0.5 ? "EARLY_SPRING" : "LONG_WINTER";
  const certainty = totalWeight ? Math.max(pEarly, 1 - pEarly) : Number.NaN;

  return { latestYear, pred, certainty, used, totalWeight };
}

function renderLeaderboard(data) {
  const table = $("leaderboard");
  if (!table) return;
  const groundhogs = Array.isArray(data?.groundhogs) ? [...data.groundhogs] : [];
  groundhogs.sort((a, b) => b.accuracy - a.accuracy);

  const rows = groundhogs.map((g, idx) => {
    const accuracy = Number.isFinite(g.accuracy) ? g.accuracy : Number.NaN;
    return `
      <tr>
        <td class="rank">${String(idx + 1).padStart(2, "0")}</td>
        <td>${g.name}</td>
        <td class="accuracy">${fmtPctValue(accuracy, 2)}</td>
      </tr>
    `;
  }).join("");

  table.innerHTML = `
    <thead>
      <tr>
        <th class="rank">Rank</th>
        <th>Groundhog</th>
        <th>Accuracy</th>
      </tr>
    </thead>
    <tbody>
      ${rows}
    </tbody>
  `;
}

async function run() {
  try {
    setStatus("Loading data...");

    const [predObj, outcomesText] = await Promise.all([
      loadJson("./data/predictions.json"),
      loadText("./data/outcomes.csv")
    ]);

    let noaaObj = null;
    try {
      noaaObj = await loadJson("./data/noaa_groundhogs.json");
    } catch (err) {
      noaaObj = NOAA_FALLBACK;
    }

    if (noaaObj?.period) {
      $("leaderboardMeta").textContent = `NOAA Heritage accuracy list (${noaaObj.period}).`;
    }
    renderLeaderboard(noaaObj);

    const outcomesRows = parseCSV(outcomesText);
    const outcomes = indexOutcomes(outcomesRows);
    const predByYear = indexPredictions(predObj);

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
      : "Prediction uses the highest-accuracy algorithm from the backtest set.";

    setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(String(err));
  }
}

run();
