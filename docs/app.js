import { parseCSV, wilsonCI } from "./lib/stats.js";
import { indexOutcomes, indexPredictions, trainWeights, predictionToOutcome } from "./lib/backtest.js";

const $ = (id) => document.getElementById(id);

const TARGET_BASE = "US_CONUS_FEBMAR_MEAN_ANOM";
const TARGET_MARCH = "US_CONUS_MAR_ANOM";
const MIN_OBS = 20;
const MIN_BACKTEST_GH = 20;
const DEFAULT_OPTS = {
  minObs: MIN_OBS,
  betaPrior: [2, 2],
  halfLifeYears: 10,
  alpha: 2.0,
  gamma: 0.5
};

const ALGORITHMS = [
  { key: "bayes", method: "bayes", label: "Bayesian mean + sample boost" },
  { key: "smooth_acc", method: "smooth_acc", label: "Smoothed accuracy" },
  { key: "exp_decay", method: "exp_decay", label: "Exponentially-decayed accuracy" },
  { key: "exp_decay_5", method: "exp_decay", label: "Exp decay (half-life 5y)", opts: { halfLifeYears: 5 } },
  { key: "exp_decay_20", method: "exp_decay", label: "Exp decay (half-life 20y)", opts: { halfLifeYears: 20 } },
  { key: "logit", method: "logit", label: "Logit weighting (flips contrarians)" },
  { key: "logit_decay", method: "logit_decay", label: "Logit weighting + decay" },
  { key: "wilson", method: "wilson", label: "Wilson confidence weighting" },
  { key: "wilson_decay", method: "wilson_decay", label: "Wilson weighting + decay" },
  { key: "zscore", method: "zscore", label: "Z-score weighting" },
  { key: "zscore_decay", method: "zscore_decay", label: "Z-score weighting + decay" },
  { key: "centered", method: "centered", label: "Centered accuracy weighting" },
  { key: "centered_decay", method: "centered_decay", label: "Centered weighting + decay" },
  { key: "logit_win5", method: "logit", label: "Logit weighting (5y window)", opts: { windowYears: 5 } },
  { key: "logit_win10", method: "logit", label: "Logit weighting (10y window)", opts: { windowYears: 10 } },
  { key: "logit_win20", method: "logit", label: "Logit weighting (20y window)", opts: { windowYears: 20 } },
  { key: "wilson_win5", method: "wilson", label: "Wilson weighting (5y window)", opts: { windowYears: 5 } },
  { key: "wilson_win10", method: "wilson", label: "Wilson weighting (10y window)", opts: { windowYears: 10 } },
  { key: "zscore_win5", method: "zscore", label: "Z-score weighting (5y window)", opts: { windowYears: 5 } },
  { key: "zscore_win10", method: "zscore", label: "Z-score weighting (10y window)", opts: { windowYears: 10 } },
  { key: "bayes_alpha1", method: "bayes", label: "Bayes (alpha 1.0)", opts: { alpha: 1.0 } },
  { key: "bayes_alpha3", method: "bayes", label: "Bayes (alpha 3.0)", opts: { alpha: 3.0 } },
  { key: "logit_alpha1", method: "logit", label: "Logit (alpha 1.0)", opts: { alpha: 1.0 } },
  { key: "logit_alpha3", method: "logit", label: "Logit (alpha 3.0)", opts: { alpha: 3.0 } },
  { key: "prior_1_1", method: "bayes", label: "Bayes prior (1,1)", opts: { betaPrior: [1, 1] } },
  { key: "prior_4_4", method: "bayes", label: "Bayes prior (4,4)", opts: { betaPrior: [4, 4] } }
];

const TOP_N_CHOICES = [5, 10, 20, 30, 40];

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

function calibrateCertainty(consensus, accuracy, backtestN) {
  if (!Number.isFinite(consensus)) return consensus;
  if (!Number.isFinite(accuracy)) return consensus;
  let accAdj = accuracy;
  if (Number.isFinite(backtestN) && backtestN > 0) {
    // Shrink with a conservative Beta(1,1) prior to avoid 100% certainty on small samples.
    accAdj = (accuracy * backtestN + 1) / (backtestN + 2);
  }
  const rawSkill = Math.max(0, Math.min(1, 2 * accAdj - 1));
  const skill = Math.sqrt(rawSkill);
  const centered = consensus - 0.5;
  const scaled = 0.5 + centered * skill;
  return Math.max(0, Math.min(1, scaled));
}

function setStatus(msg) {
  const el = $("status");
  if (el) el.textContent = msg;
}

function buildOutcomeRows(outcomesRows) {
  const rows = outcomesRows.map(r => ({ ...r }));
  const extra = [];
  for (const r of outcomesRows) {
    const target = String(r.target || "").trim();
    if (target !== TARGET_BASE) continue;
    const mar = Number.parseFloat(r.mar_anom);
    if (!Number.isFinite(mar)) continue;
    extra.push({
      year: r.year,
      target: TARGET_MARCH,
      outcome: mar > 0 ? "EARLY_SPRING" : "LONG_WINTER"
    });
  }
  return rows.concat(extra);
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

function hasMinGroundhogs(predByYear, year, minCount = MIN_BACKTEST_GH) {
  const preds = predByYear.get(year) ?? [];
  return preds.length >= minCount;
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

function predictForYear(predByYear, outcomes, target, year, mode, algo, topN) {
  const preds = predByYear.get(year) ?? [];
  const totalPreds = preds.length;
  if (!totalPreds) {
    return { pred: "", certainty: Number.NaN, used: 0, totalPreds, usedWeighted: false };
  }

  if (mode === "majority_all") {
    const res = majorityVote(preds);
    return { ...res, totalPreds, usedWeighted: false };
  }

  const methodId = algo?.method || "bayes";
  const opts = { ...DEFAULT_OPTS, ...(algo?.opts ?? {}) };
  const { weights, stats } = trainWeights(predByYear, outcomes, target, year, methodId, opts);
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

  if (mode === "champion_window") {
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

function backtestMode(predByYear, outcomes, target, mode, algo, topN) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => outcomes.has(`${target}:${y}`))
    .filter((y) => hasMinGroundhogs(predByYear, y));

  let k = 0;
  let n = 0;
  let lastYear = null;

  for (const y of years) {
    const res = predictForYear(predByYear, outcomes, target, y, mode, algo, topN);
    if (!res.pred) continue;
    n += 1;
    if (res.pred === outcomes.get(`${target}:${y}`)) k += 1;
    lastYear = y;
  }

  return { accuracy: n ? k / n : Number.NaN, backtestN: n, lastYear };
}

function pickBestMethodForMode(predByYear, outcomes, target, mode, topN) {
  const results = ALGORITHMS.map((algo, index) => {
    const backtest = backtestMode(predByYear, outcomes, target, mode, algo, topN);
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

function buildBaseCandidates(predByYear, outcomes, target, labelSuffix = "") {
  const candidates = [];

  const bestAuto = pickBestMethodForMode(predByYear, outcomes, target, "auto_weighted", null);
  if (bestAuto) {
    candidates.push({
      target,
      mode: "auto_weighted",
      algo: bestAuto.algo,
      topN: null,
      label: `Auto weighted ensemble (${bestAuto.algo.label})${labelSuffix}`,
      backtest: bestAuto
    });
  }

  for (const n of TOP_N_CHOICES) {
    const bestWeighted = pickBestMethodForMode(predByYear, outcomes, target, "topn_weighted", n);
    if (bestWeighted) {
      candidates.push({
        target,
        mode: "topn_weighted",
        algo: bestWeighted.algo,
        topN: n,
        label: `Top-${n} weighted ensemble (${bestWeighted.algo.label})${labelSuffix}`,
        backtest: bestWeighted
      });
    }
  }

  for (const n of TOP_N_CHOICES) {
    const bestMajority = pickBestMethodForMode(predByYear, outcomes, target, "topn_majority", n);
    if (bestMajority) {
      candidates.push({
        target,
        mode: "topn_majority",
        algo: bestMajority.algo,
        topN: n,
        label: `Top-${n} majority vote (${bestMajority.algo.label})${labelSuffix}`,
        backtest: bestMajority
      });
    }
  }

  const bestSingle = pickBestMethodForMode(predByYear, outcomes, target, "best_single", 1);
  if (bestSingle) {
    candidates.push({
      target,
      mode: "best_single",
      algo: bestSingle.algo,
      topN: 1,
      label: `Best single groundhog (${bestSingle.algo.label})${labelSuffix}`,
      backtest: bestSingle
    });
  }

  for (const windowYears of [10, 20, 30, 40]) {
    const champAlgo = { key: `champion_${windowYears}`, method: "bayes", opts: { windowYears } };
    const champion = backtestMode(predByYear, outcomes, target, "champion_window", champAlgo, null);
    if (Number.isFinite(champion.accuracy)) {
      candidates.push({
        target,
        mode: "champion_window",
        algo: champAlgo,
        topN: null,
        label: `Champion (${windowYears}y window)${labelSuffix}`,
        backtest: champion
      });
    }
  }

  const flipMajority = backtestMode(predByYear, outcomes, target, "flip_majority", ALGORITHMS[0], null);
  if (Number.isFinite(flipMajority.accuracy)) {
    candidates.push({
      target,
      mode: "flip_majority",
      algo: ALGORITHMS[0],
      topN: null,
      label: `Flip contrarians (Wilson filter)${labelSuffix}`,
      backtest: flipMajority
    });
  }

  const majorityAll = backtestMode(predByYear, outcomes, target, "majority_all", null, null);
  if (Number.isFinite(majorityAll.accuracy)) {
    candidates.push({
      target,
      mode: "majority_all",
      algo: null,
      topN: null,
      label: `Majority vote (all groundhogs)${labelSuffix}`,
      backtest: majorityAll
    });
  }

  return candidates;
}

function candidateKey(c) {
  return `${c.target}|${c.mode}|${c.algo?.key ?? "none"}|${c.topN ?? ""}`;
}

function getCandidatePrediction(cache, predByYear, outcomes, year, c) {
  const key = candidateKey(c);
  if (!cache.has(key)) cache.set(key, new Map());
  const byYear = cache.get(key);
  if (!byYear.has(year)) {
    byYear.set(year, predictForYear(predByYear, outcomes, c.target, year, c.mode, c.algo, c.topN));
  }
  return byYear.get(year);
}

function metaWeightedPredict(predByYear, outcomes, target, year, baseCandidates, windowYears, cache) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => y < year && outcomes.has(`${target}:${y}`));
  const windowStart = windowYears ? (year - windowYears) : null;

  let score = 0;
  let totalAbs = 0;
  let used = 0;

  for (const c of baseCandidates) {
    let k = 0;
    let n = 0;
    for (const y of years) {
      if (windowStart && y < windowStart) continue;
      const res = getCandidatePrediction(cache, predByYear, outcomes, y, c);
      if (!res?.pred) continue;
      n += 1;
      if (res.pred === outcomes.get(`${target}:${y}`)) k += 1;
    }
    if (!n) continue;
    const p = (k + 2) / (n + 4);
    const w = Math.log(p / (1 - p));

    const cur = getCandidatePrediction(cache, predByYear, outcomes, year, c);
    if (!cur?.pred) continue;
    const vote = cur.pred === "EARLY_SPRING" ? 1 : -1;
    score += w * vote;
    totalAbs += Math.abs(w);
    used += 1;
  }

  if (!totalAbs) return { pred: "", certainty: Number.NaN, used };
  const pred = score >= 0 ? "EARLY_SPRING" : "LONG_WINTER";
  const certainty = Math.min(1, Math.abs(score) / totalAbs);
  return { pred, certainty, used };
}

function metaWeightedDecayPredict(predByYear, outcomes, target, year, baseCandidates, halfLifeYears, cache) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => y < year && outcomes.has(`${target}:${y}`));
  const lambda = Math.log(2) / Math.max(1e-9, halfLifeYears);

  let score = 0;
  let totalAbs = 0;
  let used = 0;

  for (const c of baseCandidates) {
    let k = 0;
    let n = 0;
    for (const y of years) {
      const res = getCandidatePrediction(cache, predByYear, outcomes, y, c);
      if (!res?.pred) continue;
      const w = Math.exp(-lambda * (year - y));
      n += w;
      if (res.pred === outcomes.get(`${target}:${y}`)) k += w;
    }
    if (!n) continue;
    const p = (k + 2) / (n + 4);
    const w = Math.log(p / (1 - p));

    const cur = getCandidatePrediction(cache, predByYear, outcomes, year, c);
    if (!cur?.pred) continue;
    const vote = cur.pred === "EARLY_SPRING" ? 1 : -1;
    score += w * vote;
    totalAbs += Math.abs(w);
    used += 1;
  }

  if (!totalAbs) return { pred: "", certainty: Number.NaN, used };
  const pred = score >= 0 ? "EARLY_SPRING" : "LONG_WINTER";
  const certainty = Math.min(1, Math.abs(score) / totalAbs);
  return { pred, certainty, used };
}

function metaBestPredict(predByYear, outcomes, target, year, baseCandidates, windowYears, cache) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => y < year && outcomes.has(`${target}:${y}`));
  const windowStart = windowYears ? (year - windowYears) : null;

  let best = null;
  for (const c of baseCandidates) {
    let k = 0;
    let n = 0;
    for (const y of years) {
      if (windowStart && y < windowStart) continue;
      const res = getCandidatePrediction(cache, predByYear, outcomes, y, c);
      if (!res?.pred) continue;
      n += 1;
      if (res.pred === outcomes.get(`${target}:${y}`)) k += 1;
    }
    if (!n) continue;
    const acc = k / n;
    if (!best || acc > best.acc || (acc === best.acc && n > best.n)) {
      best = { candidate: c, acc, n };
    }
  }

  if (!best) return { pred: "", certainty: Number.NaN, used: 0 };
  const cur = getCandidatePrediction(cache, predByYear, outcomes, year, best.candidate);
  if (!cur?.pred) return { pred: "", certainty: Number.NaN, used: 0 };
  return { pred: cur.pred, certainty: best.acc, used: 1 };
}

function metaBestDecayPredict(predByYear, outcomes, target, year, baseCandidates, halfLifeYears, cache) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => y < year && outcomes.has(`${target}:${y}`));
  const lambda = Math.log(2) / Math.max(1e-9, halfLifeYears);

  let best = null;
  for (const c of baseCandidates) {
    let k = 0;
    let n = 0;
    for (const y of years) {
      const res = getCandidatePrediction(cache, predByYear, outcomes, y, c);
      if (!res?.pred) continue;
      const w = Math.exp(-lambda * (year - y));
      n += w;
      if (res.pred === outcomes.get(`${target}:${y}`)) k += w;
    }
    if (!n) continue;
    const acc = (k + 1) / (n + 2);
    if (!best || acc > best.acc || (acc === best.acc && n > best.n)) {
      best = { candidate: c, acc, n };
    }
  }

  if (!best) return { pred: "", certainty: Number.NaN, used: 0 };
  const cur = getCandidatePrediction(cache, predByYear, outcomes, year, best.candidate);
  if (!cur?.pred) return { pred: "", certainty: Number.NaN, used: 0 };
  return { pred: cur.pred, certainty: best.acc, used: 1 };
}

function superEnsemblePredict(predByYear, outcomes, target, year, baseCandidates, cfg, cache) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => y < year && outcomes.has(`${target}:${y}`));
  const windowStart = cfg.windowYears ? (year - cfg.windowYears) : null;
  const lambda = cfg.halfLifeYears ? Math.log(2) / Math.max(1e-9, cfg.halfLifeYears) : null;

  const scored = [];
  for (const c of baseCandidates) {
    let k = 0;
    let n = 0;
    for (const y of years) {
      if (windowStart && y < windowStart) continue;
      const res = getCandidatePrediction(cache, predByYear, outcomes, y, c);
      if (!res?.pred) continue;
      const w = lambda ? Math.exp(-lambda * (year - y)) : 1;
      n += w;
      if (res.pred === outcomes.get(`${target}:${y}`)) k += w;
    }
    if (!n) continue;
    const acc = (k + 2) / (n + 4);
    scored.push({ candidate: c, acc, n });
  }

  if (!scored.length) return { pred: "", certainty: Number.NaN, used: 0 };
  scored.sort((a, b) => {
    if (b.acc !== a.acc) return b.acc - a.acc;
    return b.n - a.n;
  });

  const topK = Math.max(1, Math.min(scored.length, cfg.topK ?? 8));
  const selected = scored.slice(0, topK);

  let score = 0;
  let totalAbs = 0;
  let used = 0;
  for (const s of selected) {
    const cur = getCandidatePrediction(cache, predByYear, outcomes, year, s.candidate);
    if (!cur?.pred) continue;
    const p = Math.min(1 - 1e-6, Math.max(1e-6, s.acc));
    const w = Math.log(p / (1 - p));
    const vote = cur.pred === "EARLY_SPRING" ? 1 : -1;
    score += w * vote;
    totalAbs += Math.abs(w);
    used += 1;
  }

  if (!totalAbs) return { pred: "", certainty: Number.NaN, used };
  const pred = score >= 0 ? "EARLY_SPRING" : "LONG_WINTER";
  const certainty = Math.min(1, Math.abs(score) / totalAbs);
  return { pred, certainty, used };
}

function superStablePredict(predByYear, outcomes, target, year, baseCandidates, cfg, cache) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => y < year && outcomes.has(`${target}:${y}`));
  const windowStart = cfg.windowYears ? (year - cfg.windowYears) : null;
  const lambda = cfg.halfLifeYears ? Math.log(2) / Math.max(1e-9, cfg.halfLifeYears) : null;

  const scored = [];
  const mid = Math.floor(years.length / 2);
  const splitYear = years[mid] ?? null;

  for (const c of baseCandidates) {
    let k = 0;
    let n = 0;
    let kEarly = 0;
    let nEarly = 0;
    let kLate = 0;
    let nLate = 0;

    for (const y of years) {
      if (windowStart && y < windowStart) continue;
      const res = getCandidatePrediction(cache, predByYear, outcomes, y, c);
      if (!res?.pred) continue;
      const w = lambda ? Math.exp(-lambda * (year - y)) : 1;
      n += w;
      if (res.pred === outcomes.get(`${target}:${y}`)) k += w;

      if (splitYear !== null && y < splitYear) {
        nEarly += w;
        if (res.pred === outcomes.get(`${target}:${y}`)) kEarly += w;
      } else {
        nLate += w;
        if (res.pred === outcomes.get(`${target}:${y}`)) kLate += w;
      }
    }
    if (!n) continue;
    const acc = (k + 2) / (n + 4);
    const accEarly = nEarly ? kEarly / nEarly : acc;
    const accLate = nLate ? kLate / nLate : acc;
    const stability = Math.max(0, 1 - Math.abs(accEarly - accLate));
    scored.push({ candidate: c, acc, stability });
  }

  if (!scored.length) return { pred: "", certainty: Number.NaN, used: 0 };
  scored.sort((a, b) => {
    if (b.acc !== a.acc) return b.acc - a.acc;
    return b.stability - a.stability;
  });

  const topK = Math.max(1, Math.min(scored.length, cfg.topK ?? 8));
  const selected = scored.slice(0, topK);

  let score = 0;
  let totalAbs = 0;
  let used = 0;
  const power = cfg.stabilityPower ?? 1.0;

  for (const s of selected) {
    const cur = getCandidatePrediction(cache, predByYear, outcomes, year, s.candidate);
    if (!cur?.pred) continue;
    const p = Math.min(1 - 1e-6, Math.max(1e-6, s.acc));
    const w = Math.log(p / (1 - p)) * Math.pow(s.stability, power);
    const vote = cur.pred === "EARLY_SPRING" ? 1 : -1;
    score += w * vote;
    totalAbs += Math.abs(w);
    used += 1;
  }

  if (!totalAbs) return { pred: "", certainty: Number.NaN, used };
  const pred = score >= 0 ? "EARLY_SPRING" : "LONG_WINTER";
  const certainty = Math.min(1, Math.abs(score) / totalAbs);
  return { pred, certainty, used };
}

function hedgePredict(predByYear, outcomes, target, year, baseCandidates, cfg, cache) {
  const superRes = superEnsemblePredict(predByYear, outcomes, target, year, baseCandidates, cfg, cache);
  if (!superRes?.pred) return { pred: "", certainty: Number.NaN, used: 0 };
  const bestSingle = predictForYear(predByYear, outcomes, target, year, "best_single", { method: "bayes" }, 1);
  const majority = predictForYear(predByYear, outcomes, target, year, "majority_all", null, null);
  if (!bestSingle?.pred || !majority?.pred) return superRes;

  const wSuper = cfg.wSuper ?? 0.6;
  const wBest = cfg.wBest ?? 0.3;
  const wMaj = cfg.wMaj ?? 0.1;
  const parts = [
    { pred: superRes.pred, w: wSuper },
    { pred: bestSingle.pred, w: wBest },
    { pred: majority.pred, w: wMaj }
  ];

  let score = 0;
  let totalAbs = 0;
  for (const part of parts) {
    const vote = part.pred === "EARLY_SPRING" ? 1 : -1;
    score += part.w * vote;
    totalAbs += Math.abs(part.w);
  }
  if (!totalAbs) return superRes;
  const pred = score >= 0 ? "EARLY_SPRING" : "LONG_WINTER";
  const certainty = Math.min(1, Math.abs(score) / totalAbs);
  return { pred, certainty, used: parts.length };
}

function superGatePredict(predByYear, outcomes, target, year, baseCandidates, cfg, gate, cache) {
  const res = superEnsemblePredict(predByYear, outcomes, target, year, baseCandidates, cfg, cache);
  if (!res.pred) return res;
  if (res.certainty >= gate) return res;

  // Fallback to best single (historical champ)
  const bestAlgo = { key: "gate_fallback_bayes", method: "bayes" };
  const fallback = predictForYear(predByYear, outcomes, target, year, "best_single", bestAlgo, 1);
  if (!fallback?.pred) return res;
  return { pred: fallback.pred, certainty: fallback.certainty, used: fallback.used, usedWeighted: false };
}

function backtestMeta(predByYear, outcomes, target, baseCandidates, windowYears, strategy) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => outcomes.has(`${target}:${y}`))
    .filter((y) => hasMinGroundhogs(predByYear, y));
  const cache = new Map();
  let k = 0;
  let n = 0;
  let lastYear = null;

  for (const y of years) {
    const res = strategy === "best"
      ? metaBestPredict(predByYear, outcomes, target, y, baseCandidates, windowYears, cache)
      : metaWeightedPredict(predByYear, outcomes, target, y, baseCandidates, windowYears, cache);
    if (!res.pred) continue;
    n += 1;
    if (res.pred === outcomes.get(`${target}:${y}`)) k += 1;
    lastYear = y;
  }

  return { accuracy: n ? k / n : Number.NaN, backtestN: n, lastYear };
}

function backtestMetaDecay(predByYear, outcomes, target, baseCandidates, halfLifeYears, strategy) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => outcomes.has(`${target}:${y}`))
    .filter((y) => hasMinGroundhogs(predByYear, y));
  const cache = new Map();
  let k = 0;
  let n = 0;
  let lastYear = null;

  for (const y of years) {
    const res = strategy === "best"
      ? metaBestDecayPredict(predByYear, outcomes, target, y, baseCandidates, halfLifeYears, cache)
      : metaWeightedDecayPredict(predByYear, outcomes, target, y, baseCandidates, halfLifeYears, cache);
    if (!res.pred) continue;
    n += 1;
    if (res.pred === outcomes.get(`${target}:${y}`)) k += 1;
    lastYear = y;
  }

  return { accuracy: n ? k / n : Number.NaN, backtestN: n, lastYear };
}

function backtestSuperEnsemble(predByYear, outcomes, target, baseCandidates, cfg) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => outcomes.has(`${target}:${y}`));
  const cache = new Map();
  let k = 0;
  let n = 0;
  let lastYear = null;

  for (const y of years) {
    const res = superEnsemblePredict(predByYear, outcomes, target, y, baseCandidates, cfg, cache);
    if (!res.pred) continue;
    n += 1;
    if (res.pred === outcomes.get(`${target}:${y}`)) k += 1;
    lastYear = y;
  }

  return { accuracy: n ? k / n : Number.NaN, backtestN: n, lastYear };
}

function backtestSuperStable(predByYear, outcomes, target, baseCandidates, cfg) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => outcomes.has(`${target}:${y}`))
    .filter((y) => hasMinGroundhogs(predByYear, y));
  const cache = new Map();
  let k = 0;
  let n = 0;
  let lastYear = null;

  for (const y of years) {
    const res = superStablePredict(predByYear, outcomes, target, y, baseCandidates, cfg, cache);
    if (!res.pred) continue;
    n += 1;
    if (res.pred === outcomes.get(`${target}:${y}`)) k += 1;
    lastYear = y;
  }

  return { accuracy: n ? k / n : Number.NaN, backtestN: n, lastYear };
}

function backtestHedge(predByYear, outcomes, target, baseCandidates, cfg) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => outcomes.has(`${target}:${y}`))
    .filter((y) => hasMinGroundhogs(predByYear, y));
  const cache = new Map();
  let k = 0;
  let n = 0;
  let lastYear = null;

  for (const y of years) {
    const res = hedgePredict(predByYear, outcomes, target, y, baseCandidates, cfg, cache);
    if (!res.pred) continue;
    n += 1;
    if (res.pred === outcomes.get(`${target}:${y}`)) k += 1;
    lastYear = y;
  }

  return { accuracy: n ? k / n : Number.NaN, backtestN: n, lastYear };
}

function backtestSuperGate(predByYear, outcomes, target, baseCandidates, cfg, gate) {
  const years = Array.from(predByYear.keys()).sort((a, b) => a - b)
    .filter((y) => outcomes.has(`${target}:${y}`))
    .filter((y) => hasMinGroundhogs(predByYear, y));
  const cache = new Map();
  let k = 0;
  let n = 0;
  let lastYear = null;

  for (const y of years) {
    const res = superGatePredict(predByYear, outcomes, target, y, baseCandidates, cfg, gate, cache);
    if (!res.pred) continue;
    n += 1;
    if (res.pred === outcomes.get(`${target}:${y}`)) k += 1;
    lastYear = y;
  }

  return { accuracy: n ? k / n : Number.NaN, backtestN: n, lastYear };
}

function buildFeatureCache(predByYear, outcomes, target, baseCandidates) {
  const allYears = Array.from(predByYear.keys()).sort((a, b) => a - b);
  const scoredYears = allYears.filter((y) => outcomes.has(`${target}:${y}`) && hasMinGroundhogs(predByYear, y));
  const cache = new Map();
  const featuresByYear = new Map();
  const labelsByYear = new Map();

  for (const y of allYears) {
    const feats = baseCandidates.map((c) => {
      const res = getCandidatePrediction(cache, predByYear, outcomes, y, c);
      if (!res?.pred) return 0;
      return res.pred === "EARLY_SPRING" ? 1 : -1;
    });
    featuresByYear.set(y, feats);
  }

  for (const y of scoredYears) {
    const out = outcomes.get(`${target}:${y}`);
    labelsByYear.set(y, out === "EARLY_SPRING" ? 1 : 0);
  }

  return { allYears, scoredYears, featuresByYear, labelsByYear, baseCandidates, target };
}

function trainLogistic(features, labels, opts = {}) {
  const steps = opts.steps ?? 200;
  const lr = opts.lr ?? 0.2;
  const l2 = opts.l2 ?? 0.01;
  const m = features[0]?.length ?? 0;
  if (!m) return null;

  let w = new Array(m).fill(0);
  let b = 0;
  const n = features.length;
  const sampleWeights = opts.sampleWeights ?? null;
  const weightSum = sampleWeights ? sampleWeights.reduce((s, x) => s + x, 0) : n;
  const invN = weightSum ? 1 / weightSum : 0;

  for (let iter = 0; iter < steps; iter++) {
    const gradW = new Array(m).fill(0);
    let gradB = 0;

    for (let i = 0; i < n; i++) {
      const x = features[i];
      const wi = sampleWeights ? sampleWeights[i] : 1;
      let z = b;
      for (let j = 0; j < m; j++) z += w[j] * x[j];
      const p = 1 / (1 + Math.exp(-z));
      const diff = (p - labels[i]) * wi;
      for (let j = 0; j < m; j++) gradW[j] += diff * x[j];
      gradB += diff;
    }

    for (let j = 0; j < m; j++) {
      gradW[j] = gradW[j] * invN + l2 * w[j];
      w[j] -= lr * gradW[j];
    }
    gradB *= invN;
    b -= lr * gradB;
  }

  return { w, b };
}

function stackedPredict(featureCache, year, windowYears, opts = {}) {
  const trainYears = featureCache.scoredYears.filter((y) => y < year && (!windowYears || y >= year - windowYears));
  if (trainYears.length < (opts.minTrain ?? 12)) {
    return { pred: "", certainty: Number.NaN, used: 0 };
  }

  const X = trainYears.map((y) => featureCache.featuresByYear.get(y));
  const y = trainYears.map((y) => featureCache.labelsByYear.get(y));
  let sampleWeights = null;
  if (opts.decayHalfLife) {
    const lambda = Math.log(2) / Math.max(1e-9, opts.decayHalfLife);
    sampleWeights = trainYears.map((y) => Math.exp(-lambda * (year - y)));
  }
  const model = trainLogistic(X, y, { ...opts, sampleWeights });
  if (!model) return { pred: "", certainty: Number.NaN, used: 0 };

  const xCur = featureCache.featuresByYear.get(year);
  if (!xCur) return { pred: "", certainty: Number.NaN, used: 0 };

  let z = model.b;
  for (let j = 0; j < model.w.length; j++) z += model.w[j] * xCur[j];
  const p = 1 / (1 + Math.exp(-z));
  const pred = p >= 0.5 ? "EARLY_SPRING" : "LONG_WINTER";
  const certainty = Math.abs(p - 0.5) * 2;
  const used = xCur.reduce((c, v) => c + (v !== 0 ? 1 : 0), 0);
  return { pred, certainty, used };
}

function backtestStacked(featureCache, windowYears, opts = {}) {
  let k = 0;
  let n = 0;
  let lastYear = null;

  for (const y of featureCache.scoredYears) {
    const res = stackedPredict(featureCache, y, windowYears, opts);
    if (!res.pred) continue;
    const actual = featureCache.labelsByYear.get(y) === 1 ? "EARLY_SPRING" : "LONG_WINTER";
    n += 1;
    if (res.pred === actual) k += 1;
    lastYear = y;
  }

  return { accuracy: n ? k / n : Number.NaN, backtestN: n, lastYear };
}

function pickBestOverall(predByYear, outcomes) {
  const candidates = [];

  const baseFebMar = buildBaseCandidates(predByYear, outcomes, TARGET_BASE, "");
  const baseMarch = buildBaseCandidates(predByYear, outcomes, TARGET_MARCH, " (March-only)");
  candidates.push(...baseFebMar, ...baseMarch);

  for (const [target, baseCandidates, labelSuffix] of [
    [TARGET_BASE, baseFebMar, ""],
    [TARGET_MARCH, baseMarch, " (March-only)"]
  ]) {
    const featureCache = buildFeatureCache(predByYear, outcomes, target, baseCandidates);
    for (const windowYears of [15, 25, 40]) {
      const metaWeighted = backtestMeta(predByYear, outcomes, target, baseCandidates, windowYears, "weighted");
      if (Number.isFinite(metaWeighted.accuracy)) {
        candidates.push({
          target,
          mode: "meta_weighted",
          algo: { key: `meta_weighted_${windowYears}_${target}` },
          topN: null,
          label: `Meta ensemble (${windowYears}y window)${labelSuffix}`,
          backtest: metaWeighted,
          meta: { strategy: "weighted", windowYears, baseCandidates, target }
        });
      }

      const metaBest = backtestMeta(predByYear, outcomes, target, baseCandidates, windowYears, "best");
      if (Number.isFinite(metaBest.accuracy)) {
        candidates.push({
          target,
          mode: "meta_best",
          algo: { key: `meta_best_${windowYears}_${target}` },
          topN: null,
          label: `Meta best-of (${windowYears}y window)${labelSuffix}`,
          backtest: metaBest,
          meta: { strategy: "best", windowYears, baseCandidates, target }
        });
      }

      const stacked = backtestStacked(featureCache, windowYears, { steps: 220, lr: 0.2, l2: 0.05, minTrain: 12 });
      if (Number.isFinite(stacked.accuracy)) {
        candidates.push({
          target,
          mode: "stacked_cv",
          algo: { key: `stacked_${windowYears}_${target}` },
          topN: null,
          label: `Stacked logistic (${windowYears}y window)${labelSuffix}`,
          backtest: stacked,
          meta: { type: "stacked", windowYears, featureCache, opts: { steps: 220, lr: 0.2, l2: 0.05, minTrain: 12 } }
        });
      }
    }

    for (const halfLifeYears of [5, 10, 20]) {
      const metaWeightedDecay = backtestMetaDecay(predByYear, outcomes, target, baseCandidates, halfLifeYears, "weighted");
      if (Number.isFinite(metaWeightedDecay.accuracy)) {
        candidates.push({
          target,
          mode: "meta_weighted_decay",
          algo: { key: `meta_weighted_decay_${halfLifeYears}_${target}` },
          topN: null,
          label: `Meta ensemble (half-life ${halfLifeYears}y)${labelSuffix}`,
          backtest: metaWeightedDecay,
          meta: { strategy: "weighted_decay", halfLifeYears, baseCandidates, target }
        });
      }

      const metaBestDecay = backtestMetaDecay(predByYear, outcomes, target, baseCandidates, halfLifeYears, "best");
      if (Number.isFinite(metaBestDecay.accuracy)) {
        candidates.push({
          target,
          mode: "meta_best_decay",
          algo: { key: `meta_best_decay_${halfLifeYears}_${target}` },
          topN: null,
          label: `Meta best-of (half-life ${halfLifeYears}y)${labelSuffix}`,
          backtest: metaBestDecay,
          meta: { strategy: "best_decay", halfLifeYears, baseCandidates, target }
        });
      }

      const stackedDecay = backtestStacked(featureCache, null, { steps: 240, lr: 0.2, l2: 0.05, minTrain: 12, decayHalfLife: halfLifeYears });
      if (Number.isFinite(stackedDecay.accuracy)) {
        candidates.push({
          target,
          mode: "stacked_decay",
          algo: { key: `stacked_decay_${halfLifeYears}_${target}` },
          topN: null,
          label: `Stacked logistic (half-life ${halfLifeYears}y)${labelSuffix}`,
          backtest: stackedDecay,
          meta: { type: "stacked", windowYears: null, featureCache, opts: { steps: 240, lr: 0.2, l2: 0.05, minTrain: 12, decayHalfLife: halfLifeYears } }
        });
      }
    }

    for (const cfg of [
      // Aggressive, recency-biased configs
      { windowYears: 12, topK: 3 },
      { windowYears: 18, topK: 5 },
      { windowYears: 20, topK: 5 },
      { halfLifeYears: 3, topK: 7 },
      { halfLifeYears: 5, topK: 3 },
      { halfLifeYears: 7, topK: 5 },
      // Moderate fallbacks
      { windowYears: 25, topK: 8 },
      { halfLifeYears: 10, topK: 8 }
    ]) {
      const superBt = backtestSuperEnsemble(predByYear, outcomes, target, baseCandidates, cfg);
      if (Number.isFinite(superBt.accuracy)) {
        const labelParts = [];
        if (cfg.windowYears) labelParts.push(`${cfg.windowYears}y window`);
        if (cfg.halfLifeYears) labelParts.push(`half-life ${cfg.halfLifeYears}y`);
        labelParts.push(`top-${cfg.topK}`);
        candidates.push({
          target,
          mode: "super_ensemble",
          algo: { key: `super_${cfg.windowYears ?? "hl" + cfg.halfLifeYears}_${cfg.topK}_${target}` },
          topN: null,
          label: `Super ensemble (${labelParts.join(", ")})${labelSuffix}`,
          backtest: superBt,
          meta: { type: "super", cfg, baseCandidates, target }
        });
      }
    }

    // Stability-penalized super-ensemble
    for (const cfg of [
      { halfLifeYears: 3, topK: 7, stabilityPower: 1.0 },
      { windowYears: 20, topK: 5, stabilityPower: 1.0 }
    ]) {
      const stable = backtestSuperStable(predByYear, outcomes, target, baseCandidates, cfg);
      if (Number.isFinite(stable.accuracy)) {
        const labelParts = [];
        if (cfg.windowYears) labelParts.push(`${cfg.windowYears}y window`);
        if (cfg.halfLifeYears) labelParts.push(`half-life ${cfg.halfLifeYears}y`);
        labelParts.push(`top-${cfg.topK}`);
        candidates.push({
          target,
          mode: "super_stable",
          algo: { key: `super_stable_${cfg.windowYears ?? "hl" + cfg.halfLifeYears}_${cfg.topK}_${target}` },
          topN: null,
          label: `Super stable (${labelParts.join(", ")})${labelSuffix}`,
          backtest: stable,
          meta: { type: "super_stable", cfg, baseCandidates, target }
        });
      }
    }

    // Hedge ensemble: super + best single + majority
    for (const cfg of [
      { halfLifeYears: 3, topK: 7, wSuper: 0.6, wBest: 0.3, wMaj: 0.1 }
    ]) {
      const hedge = backtestHedge(predByYear, outcomes, target, baseCandidates, cfg);
      if (Number.isFinite(hedge.accuracy)) {
        candidates.push({
          target,
          mode: "hedge",
          algo: { key: `hedge_${cfg.halfLifeYears ?? cfg.windowYears}_${cfg.topK}_${target}` },
          topN: null,
          label: `Hedge ensemble (super/best/majority)${labelSuffix}`,
          backtest: hedge,
          meta: { type: "hedge", cfg, baseCandidates, target }
        });
      }
    }

    // Gated super-ensemble: only trust if confidence passes threshold, else best single.
    for (const gateCfg of [
      { cfg: { halfLifeYears: 5, topK: 3 }, gate: 0.55 }
    ]) {
      const gated = backtestSuperGate(predByYear, outcomes, target, baseCandidates, gateCfg.cfg, gateCfg.gate);
      if (Number.isFinite(gated.accuracy)) {
        candidates.push({
          target,
          mode: "super_gate",
          algo: { key: `super_gate_${gateCfg.cfg.halfLifeYears}_${gateCfg.cfg.topK}_${gateCfg.gate}_${target}` },
          topN: null,
          label: `Super ensemble gate (half-life ${gateCfg.cfg.halfLifeYears}y, top-${gateCfg.cfg.topK}, gate ${Math.round(gateCfg.gate * 100)}%)${labelSuffix}`,
          backtest: gated,
          meta: { type: "super_gate", cfg: gateCfg.cfg, gate: gateCfg.gate, baseCandidates, target }
        });
      }
    }
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

function computeMetaNowcast(predByYear, outcomes, meta) {
  const years = Array.from(predByYear.keys());
  if (!years.length) return null;
  const latestYear = Math.max(...years);
  const preds = predByYear.get(latestYear) ?? [];
  const cache = new Map();
  let res;
  if (meta.strategy === "best") {
    res = metaBestPredict(predByYear, outcomes, meta.target, latestYear, meta.baseCandidates, meta.windowYears, cache);
  } else if (meta.strategy === "best_decay") {
    res = metaBestDecayPredict(predByYear, outcomes, meta.target, latestYear, meta.baseCandidates, meta.halfLifeYears, cache);
  } else if (meta.strategy === "weighted_decay") {
    res = metaWeightedDecayPredict(predByYear, outcomes, meta.target, latestYear, meta.baseCandidates, meta.halfLifeYears, cache);
  } else {
    res = metaWeightedPredict(predByYear, outcomes, meta.target, latestYear, meta.baseCandidates, meta.windowYears, cache);
  }

  return {
    latestYear,
    pred: res.pred,
    certainty: res.certainty,
    used: preds.length,
    totalPreds: preds.length,
    usedWeighted: meta.strategy === "weighted" || meta.strategy === "weighted_decay"
  };
}

function computeStackedNowcast(predByYear, meta) {
  const years = Array.from(predByYear.keys());
  if (!years.length) return null;
  const latestYear = Math.max(...years);
  const preds = predByYear.get(latestYear) ?? [];
  const res = stackedPredict(meta.featureCache, latestYear, meta.windowYears, meta.opts);
  return {
    latestYear,
    pred: res.pred,
    certainty: res.certainty,
    used: preds.length,
    totalPreds: preds.length,
    usedWeighted: true
  };
}

function computeSuperNowcast(predByYear, outcomes, meta) {
  const years = Array.from(predByYear.keys());
  if (!years.length) return null;
  const latestYear = Math.max(...years);
  const preds = predByYear.get(latestYear) ?? [];
  const cache = new Map();
  const res = superEnsemblePredict(predByYear, outcomes, meta.target, latestYear, meta.baseCandidates, meta.cfg, cache);
  return {
    latestYear,
    pred: res.pred,
    certainty: res.certainty,
    used: preds.length,
    totalPreds: preds.length,
    usedWeighted: true
  };
}

function computeSuperGateNowcast(predByYear, outcomes, meta) {
  const years = Array.from(predByYear.keys());
  if (!years.length) return null;
  const latestYear = Math.max(...years);
  const preds = predByYear.get(latestYear) ?? [];
  const cache = new Map();
  const res = superGatePredict(predByYear, outcomes, meta.target, latestYear, meta.baseCandidates, meta.cfg, meta.gate, cache);
  return {
    latestYear,
    pred: res.pred,
    certainty: res.certainty,
    used: preds.length,
    totalPreds: preds.length,
    usedWeighted: true
  };
}

function computeSuperStableNowcast(predByYear, outcomes, meta) {
  const years = Array.from(predByYear.keys());
  if (!years.length) return null;
  const latestYear = Math.max(...years);
  const preds = predByYear.get(latestYear) ?? [];
  const cache = new Map();
  const res = superStablePredict(predByYear, outcomes, meta.target, latestYear, meta.baseCandidates, meta.cfg, cache);
  return {
    latestYear,
    pred: res.pred,
    certainty: res.certainty,
    used: preds.length,
    totalPreds: preds.length,
    usedWeighted: true
  };
}

function computeHedgeNowcast(predByYear, outcomes, meta) {
  const years = Array.from(predByYear.keys());
  if (!years.length) return null;
  const latestYear = Math.max(...years);
  const preds = predByYear.get(latestYear) ?? [];
  const cache = new Map();
  const res = hedgePredict(predByYear, outcomes, meta.target, latestYear, meta.baseCandidates, meta.cfg, cache);
  return {
    latestYear,
    pred: res.pred,
    certainty: res.certainty,
    used: preds.length,
    totalPreds: preds.length,
    usedWeighted: true
  };
}

function computeNowcast(predByYear, outcomes, target, mode, algo, topN) {
  const years = Array.from(predByYear.keys());
  if (!years.length) return null;
  const latestYear = Math.max(...years);
  const res = predictForYear(predByYear, outcomes, target, latestYear, mode, algo, topN);
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
    const actual = outcomes.get(`${TARGET_BASE}:${year}`);
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
    const outcomes = indexOutcomes(buildOutcomeRows(outcomesRows));
    const predByYear = indexPredictions(predObj);
    const nameBySlug = new Map((groundhogDir?.groundhogs ?? []).map(g => [g.slug, g.name || g.slug]));

    const leaderboardRows = computeLeaderboard(predByYear, outcomes, groundhogDir);
    renderLeaderboard(leaderboardRows);

    const scoredYears = Array.from(predByYear.keys())
      .filter((y) => outcomes.has(`${TARGET_BASE}:${y}`) && hasMinGroundhogs(predByYear, y));
    const minYear = scoredYears.length ? Math.min(...scoredYears) : null;
    const maxYear = scoredYears.length ? Math.max(...scoredYears) : null;
    const leaderboardMeta = scoredYears.length
      ? `Computed from groundhog-day.com predictions vs NOAA Climate-at-a-Glance outcomes (CONUS Feb+Mar). Min obs: ${MIN_OBS}. Min groundhogs/year: ${MIN_BACKTEST_GH}. Scored years: ${minYear}–${maxYear}.`
      : `Computed from groundhog-day.com predictions vs NOAA Climate-at-a-Glance outcomes (CONUS Feb+Mar). Min obs: ${MIN_OBS}. Min groundhogs/year: ${MIN_BACKTEST_GH}.`;
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

    const nowcast = chosen.meta
      ? (chosen.meta.type === "stacked"
        ? computeStackedNowcast(predByYear, chosen.meta)
        : chosen.meta.type === "super_gate"
          ? computeSuperGateNowcast(predByYear, outcomes, chosen.meta)
          : chosen.meta.type === "super_stable"
            ? computeSuperStableNowcast(predByYear, outcomes, chosen.meta)
            : chosen.meta.type === "hedge"
              ? computeHedgeNowcast(predByYear, outcomes, chosen.meta)
          : chosen.meta.type === "super"
            ? computeSuperNowcast(predByYear, outcomes, chosen.meta)
            : computeMetaNowcast(predByYear, outcomes, chosen.meta))
      : computeNowcast(predByYear, outcomes, chosen.target, chosen.mode, chosen.algo, chosen.topN);
    if (!nowcast) {
      setStatus("No prediction data available.");
      return;
    }

    const indicator = $("indicator");
    indicator.textContent = outcomeLabel(nowcast.pred);
    document.body.dataset.outcome = nowcast.pred;

    const displayCertainty = nowcast.certainty;
    const calibratedCertainty = calibrateCertainty(nowcast.certainty, chosen.backtest.accuracy, chosen.backtest.backtestN);
    $("certainty").textContent = fmtPct(displayCertainty, 1);
    $("algoAccuracy").textContent = fmtPct(chosen.backtest.accuracy);
    $("callYear").textContent = `Forecast for ${nowcast.latestYear}`;
    $("algoName").textContent = chosen.label || "—";
    $("predictionYear").textContent = `${nowcast.latestYear}`;
    if (nowcast.totalPreds) {
      $("voterCount").textContent = `${nowcast.totalPreds} total`;
    } else {
      $("voterCount").textContent = `${nowcast.used}`;
    }

    let metaText = "Algorithm selected automatically by backtest accuracy.";
    if (chosen.target === TARGET_MARCH) {
      metaText = "Algorithm selected automatically by backtest accuracy (March-only outcomes).";
    }

    if ((chosen.mode === "best_single" || chosen.mode === "champion_window") && nowcast.selectedSlug) {
      const name = nameBySlug.get(nowcast.selectedSlug) || nowcast.selectedSlug;
      const accText = Number.isFinite(nowcast.selectedAcc) ? fmtPct(nowcast.selectedAcc) : "—";
      metaText = `Selected ${name} (${accText} historical accuracy, n=${nowcast.selectedN}).`;
    }

    if ((chosen.mode === "auto_weighted" || chosen.mode === "topn_weighted") && !nowcast.usedWeighted) {
      metaText = "Weights unavailable for this year — using simple majority vote.";
    }

    metaText += ` Calibrated confidence: ${fmtPct(calibratedCertainty, 1)} (scaled by historical accuracy).`;
    $("meta").textContent = metaText;

    if (!isSample) setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(String(err));
  }
}

run();
