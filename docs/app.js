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
  return `${x.toFixed(digits)}%`;
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
        <td class="accuracy">${fmtPctValue(accuracy)}</td>
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

    const [predObj, outcomesText, noaaObj] = await Promise.all([
      loadJson("./data/predictions.json"),
      loadText("./data/outcomes.csv"),
      loadJson("./data/noaa_groundhogs.json")
    ]);

    if (noaaObj?.period) {
      $("leaderboardMeta").textContent = `NOAA Heritage accuracy list (${noaaObj.period}).`;
    }

    renderLeaderboard(noaaObj);

    const outcomesRows = parseCSV(outcomesText);
    const outcomes = indexOutcomes(outcomesRows);
    const predByYear = indexPredictions(predObj);

    const results = evaluateAlgorithms(predByYear, outcomes);
    const best = pickBestAlgorithm(results);

    if (!best) {
      setStatus("No backtest data available.");
      return;
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
    $("algoName").textContent = best.algo.label;
    $("algoAcc").textContent = `Backtest accuracy: ${fmtPct(best.accuracy)} (n=${best.backtestN}, through ${best.lastYear})`;
    $("predictionYear").textContent = `${nowcast.latestYear}`;
    $("voterCount").textContent = `${nowcast.used}`;
    $("meta").textContent = "Prediction uses the highest-accuracy algorithm from the backtest set.";

    setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(String(err));
  }
}

run();
