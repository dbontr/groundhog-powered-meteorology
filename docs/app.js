import { parseCSV, wilsonCI } from "./lib/stats.js";
import { indexOutcomes, indexPredictions, trainWeights, predictionToOutcome } from "./lib/backtest.js";

const $ = (id) => document.getElementById(id);

const TARGET = "US_CONUS_FEBMAR_MEAN_ANOM";
const MIN_OBS = 20;
const DEFAULT_OPTS = {
  minObs: MIN_OBS,
  betaPrior: [2, 2],
  halfLifeYears: 10,
  alpha: 2.0,
  gamma: 0.5
};

const ALGORITHMS = [
  { id: "bayes", label: "Bayesian mean + sample boost" },
  { id: "smooth_acc", label: "Smoothed accuracy" },
  { id: "exp_decay", label: "Exponentially-decayed accuracy" },
  { id: "logit", label: "Logit weighting (flips contrarians)" },
  { id: "logit_decay", label: "Logit weighting + decay" },
  { id: "wilson", label: "Wilson confidence weighting" },
  { id: "wilson_decay", label: "Wilson weighting + decay" }
];

const TOP_N_CHOICES = [5, 10, 20];

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

function majorityVote(preds, allowed = null) {
  let early = 0;
  let late = 0;
  for (const p of preds) {
    if (allowed && !allowed.has(p.groundhogSlug)) continue;
    const out = predictionToOutcome(!!p.shadow);
    if (out === "EARLY_SPRING") early += 1;
    else late += 1;
  }
  const used = early + late;
  if (!used) return { pred: "", certainty: Number.NaN, used };
  const pred = early >= late ? "EARLY_SPRING" : "LONG_WINTER";
  const certainty = Math.max(early, late) / used;
  return { pred, certainty, used };
}

function majorityVoteWithFlips(preds, stats, minObs) {
  let early = 0;
  let late = 0;
  for (const p of preds) {
    const slug = p.groundhogSlug;
    const baseOut = predictionToOutcome(!!p.shadow);
    let out = baseOut;

    const s = stats?.get(slug);
    if (s && s.n >= minObs) {
      const ci = wilsonCI(s.k, s.n);
      if (ci.hi < 0.5) {
        out = (baseOut === "EARLY_SPRING") ? "LONG_WINTER" : "EARLY_SPRING";
      }
    }

    if (out === "EARLY_SPRING") early += 1;
    else late += 1;
  }
  const used = early + late;
  if (!used) return { pred: "", certainty: Number.NaN, used };
  const pred = early >= late ? "EARLY_SPRING" : "LONG_WINTER";
  const certainty = Math.max(early, late) / used;
  return { pred, certainty, used };
}

function rankStats(stats) {
  const arr = [];
  for (const [slug, s] of stats ?? []) {
    if (!s || !Number.isFinite(s.n) || s.n <= 0) continue;
    arr.push({
      slug,
      acc: s.k / s.n,
      n: s.n
    });
  }
  arr.sort((a, b) => {
    if (b.acc !== a.acc) return b.acc - a.acc;
    if (b.n !== a.n) return b.n - a.n;
    return a.slug.localeCompare(b.slug);
  });
  return arr;
}

function weightedVote(preds, weights, topN = null) {
  const items = preds
    .map((p) => ({ p, w: weights.get(p.groundhogSlug) || 0 }))
    .filter((x) => x.w !== 0);

  if (topN) {
    items.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    items.length = Math.min(items.length, topN);
  }

  let totalAbs = 0;
  let score = 0;
  for (const { p, w } of items) {
    const out = predictionToOutcome(!!p.shadow);
    const vote = (out === "EARLY_SPRING") ? 1 : -1;
    totalAbs += Math.abs(w);
    score += w * vote;
  }

  if (!totalAbs) return { pred: "", certainty: Number.NaN, used: 0, usedWeighted: false };

  const pred = score >= 0 ? "EARLY_SPRING" : "LONG_WINTER";
  const certainty = Math.min(1, Math.abs(score) / totalAbs);
  return { pred, certainty, used: items.length, usedWeighted: true };
}

function predictForYear(predByYear, outcomes, year, mode, method, topN) {
  const preds = predByYear.get(year) ?? [];
  const totalPreds = preds.length;
  if (!totalPreds) {
    return { pred: "", certainty: Number.NaN, used: 0, totalPreds, usedWeighted: false };
  }

  if (mode === "majority_all") {
    const res = majorityVote(preds);
    return { ...res, totalPreds, usedWeighted: false };
  }

  const methodId = method || "bayes";
  const { weights, stats } = trainWeights(predByYear, outcomes, TARGET, year, methodId, DEFAULT_OPTS);
  const predBySlug = new Map(preds.map((p) => [p.groundhogSlug, p]));

  if (mode === "auto_weighted") {
    const res = weightedVote(preds, weights, null);
    if (!res.used) {
      const fallback = majorityVote(preds);
      return { ...fallback, totalPreds, usedWeighted: false };
    }
    return { ...res, totalPreds };
  }

  if (mode === "topn_weighted") {
    const res = weightedVote(preds, weights, topN);
    if (!res.used) {
      const fallback = majorityVote(preds);
      return { ...fallback, totalPreds, usedWeighted: false };
    }
    return { ...res, totalPreds };
  }

  if (mode === "topn_majority") {
    const ranked = rankStats(stats);
    const allowed = new Set();
    for (const r of ranked) {
      if (predBySlug.has(r.slug)) allowed.add(r.slug);
      if (allowed.size >= topN) break;
    }
    const res = majorityVote(preds, allowed);
    if (!res.used) {
      const fallback = majorityVote(preds);
      return { ...fallback, totalPreds, usedWeighted: false };
    }
    return { ...res, totalPreds, usedWeighted: false };
  }

  if (mode === "best_single") {
    const ranked = rankStats(stats);
    for (const r of ranked) {
      const p = predBySlug.get(r.slug);
      if (!p) continue;
      const pred = predictionToOutcome(!!p.shadow);
      return {
        pred,
        certainty: r.acc,
        used: 1,
        totalPreds,
        usedWeighted: false,
        selectedSlug: r.slug,
        selectedAcc: r.acc,
        selectedN: r.n
      };
    }
    const fallback = majorityVote(preds);
    return { ...fallback, totalPreds, usedWeighted: false };
  }

  if (mode === "flip_majority") {
    const res = majorityVoteWithFlips(preds, stats, MIN_OBS);
    if (!res.used) {
      const fallback = majorityVote(preds);
      return { ...fallback, totalPreds, usedWeighted: false };
    }
    return { ...res, totalPreds, usedWeighted: false };
  }

  const fallback = majorityVote(preds);
  return { ...fallback, totalPreds, usedWeighted: false };
}

function backtestMode(predByYear, outcomes, mode, method, topN) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => outcomes.has(`${TARGET}:${y}`));

  let k = 0;
  let n = 0;
  let lastYear = null;

  for (const y of years) {
    const res = predictForYear(predByYear, outcomes, y, mode, method, topN);
    if (!res.pred) continue;
    n += 1;
    if (res.pred === outcomes.get(`${TARGET}:${y}`)) k += 1;
    lastYear = y;
  }

  return { accuracy: n ? k / n : Number.NaN, backtestN: n, lastYear };
}

function pickBestMethodForMode(predByYear, outcomes, mode, topN) {
  const results = ALGORITHMS.map((algo, index) => {
    const backtest = backtestMode(predByYear, outcomes, mode, algo.id, topN);
    return { algo, order: index, ...backtest };
  });
  const viable = results.filter((r) => Number.isFinite(r.accuracy));
  if (!viable.length) return null;
  viable.sort((a, b) => {
    if (b.accuracy !== a.accuracy) return b.accuracy - a.accuracy;
    if (b.backtestN !== a.backtestN) return b.backtestN - a.backtestN;
    return a.order - b.order;
  });
  return viable[0];
}

function pickBestOverall(predByYear, outcomes) {
  const candidates = [];

  const bestAuto = pickBestMethodForMode(predByYear, outcomes, "auto_weighted", null);
  if (bestAuto) {
    candidates.push({
      mode: "auto_weighted",
      method: bestAuto.algo.id,
      topN: null,
      label: `Auto weighted ensemble (${bestAuto.algo.label})`,
      backtest: bestAuto
    });
  }

  for (const n of TOP_N_CHOICES) {
    const bestWeighted = pickBestMethodForMode(predByYear, outcomes, "topn_weighted", n);
    if (bestWeighted) {
      candidates.push({
        mode: "topn_weighted",
        method: bestWeighted.algo.id,
        topN: n,
        label: `Top-${n} weighted ensemble (${bestWeighted.algo.label})`,
        backtest: bestWeighted
      });
    }
  }

  for (const n of TOP_N_CHOICES) {
    const bestMajority = pickBestMethodForMode(predByYear, outcomes, "topn_majority", n);
    if (bestMajority) {
      candidates.push({
        mode: "topn_majority",
        method: bestMajority.algo.id,
        topN: n,
        label: `Top-${n} majority vote (${bestMajority.algo.label})`,
        backtest: bestMajority
      });
    }
  }

  const bestSingle = pickBestMethodForMode(predByYear, outcomes, "best_single", 1);
  if (bestSingle) {
    candidates.push({
      mode: "best_single",
      method: bestSingle.algo.id,
      topN: 1,
      label: `Best single groundhog (${bestSingle.algo.label})`,
      backtest: bestSingle
    });
  }

  const flipMajority = backtestMode(predByYear, outcomes, "flip_majority", "bayes", null);
  if (Number.isFinite(flipMajority.accuracy)) {
    candidates.push({
      mode: "flip_majority",
      method: "bayes",
      topN: null,
      label: "Flip contrarians (Wilson filter)",
      backtest: flipMajority
    });
  }

  const majorityAll = backtestMode(predByYear, outcomes, "majority_all", null, null);
  if (Number.isFinite(majorityAll.accuracy)) {
    candidates.push({
      mode: "majority_all",
      method: null,
      topN: null,
      label: "Majority vote (all groundhogs)",
      backtest: majorityAll
    });
  }

  const viable = candidates.filter(c => Number.isFinite(c.backtest?.accuracy));
  if (!viable.length) return null;

  viable.sort((a, b) => {
    if (b.backtest.accuracy !== a.backtest.accuracy) return b.backtest.accuracy - a.backtest.accuracy;
    if (b.backtest.backtestN !== a.backtest.backtestN) return b.backtest.backtestN - a.backtest.backtestN;
    return a.label.localeCompare(b.label);
  });

  return viable[0];
}

function computeNowcast(predByYear, outcomes, mode, method, topN) {
  const years = Array.from(predByYear.keys());
  if (!years.length) return null;
  const latestYear = Math.max(...years);
  const res = predictForYear(predByYear, outcomes, latestYear, mode, method, topN);
  return { latestYear, ...res };
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
  })).filter(r => r.n >= MIN_OBS);

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
    const nameBySlug = new Map((groundhogDir?.groundhogs ?? []).map(g => [g.slug, g.name || g.slug]));

    const leaderboardRows = computeLeaderboard(predByYear, outcomes, groundhogDir);
    renderLeaderboard(leaderboardRows);

    const scoredYears = Array.from(predByYear.keys()).filter((y) => outcomes.has(`${TARGET}:${y}`));
    const minYear = scoredYears.length ? Math.min(...scoredYears) : null;
    const maxYear = scoredYears.length ? Math.max(...scoredYears) : null;
    const leaderboardMeta = scoredYears.length
      ? `Computed from groundhog-day.com predictions vs NOAA Climate-at-a-Glance outcomes (CONUS Feb+Mar). Min obs: ${MIN_OBS}. Scored years: ${minYear}–${maxYear}.`
      : `Computed from groundhog-day.com predictions vs NOAA Climate-at-a-Glance outcomes (CONUS Feb+Mar). Min obs: ${MIN_OBS}.`;
    $("leaderboardMeta").textContent = leaderboardMeta;

    const isSample = String(predObj.updatedAt || "").includes("SAMPLE");
    if (isSample) {
      setStatus("Sample data loaded — run python3 scripts/update_predictions.py for the full leaderboard.");
    }

    const chosen = pickBestOverall(predByYear, outcomes);
    if (!chosen) {
      setStatus("Backtest unavailable — no algorithm could be evaluated.");
      return;
    }

    const nowcast = computeNowcast(predByYear, outcomes, chosen.mode, chosen.method, chosen.topN);
    if (!nowcast) {
      setStatus("No prediction data available.");
      return;
    }

    const indicator = $("indicator");
    indicator.textContent = outcomeLabel(nowcast.pred);
    document.body.dataset.outcome = nowcast.pred;

    $("certainty").textContent = fmtPct(nowcast.certainty, 1);
    $("algoAccuracy").textContent = fmtPct(chosen.backtest.accuracy);
    $("callYear").textContent = `Forecast for ${nowcast.latestYear}`;
    $("algoName").textContent = chosen.label || "—";
    $("predictionYear").textContent = `${nowcast.latestYear}`;
    if (nowcast.totalPreds) {
      const usedText = nowcast.used !== nowcast.totalPreds ? ` (${nowcast.used} used)` : "";
      $("voterCount").textContent = `${nowcast.totalPreds} total${usedText}`;
    } else {
      $("voterCount").textContent = `${nowcast.used}`;
    }

    let metaText = "Algorithm selected automatically by backtest accuracy.";

    if (chosen.mode === "best_single" && nowcast.selectedSlug) {
      const name = nameBySlug.get(nowcast.selectedSlug) || nowcast.selectedSlug;
      const accText = Number.isFinite(nowcast.selectedAcc) ? fmtPct(nowcast.selectedAcc) : "—";
      metaText = `Selected ${name} (${accText} historical accuracy, n=${nowcast.selectedN}).`;
    }

    if ((chosen.mode === "auto_weighted" || chosen.mode === "topn_weighted") && !nowcast.usedWeighted) {
      metaText = "Weights unavailable for this year — using simple majority vote.";
    }

    $("meta").textContent = metaText;

    if (!isSample) setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(String(err));
  }
}

run();
