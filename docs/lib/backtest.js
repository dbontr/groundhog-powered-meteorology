import { wilsonCI, clamp } from "./stats.js";

const ES = "EARLY_SPRING";
const LW = "LONG_WINTER";

export function normalizeOutcome(x) {
  if (!x) return "";
  const s = String(x).trim().toUpperCase();
  if (s === "EARLY_SPRING" || s === "ES") return ES;
  // Scoring is binary, matching the groundhog prediction format.
  if (s === "LONG_WINTER" || s === "LW" || s === "MORE_WINTER") return LW;
  return "";
}

export function predictionToOutcome(shadowBool) {
  return shadowBool ? LW : ES;
}

export function indexOutcomes(outcomeRows) {
  // expects: year,target,outcome (plus optional columns)
  const m = new Map();
  for (const r of outcomeRows) {
    const year = +r.year;
    if (!Number.isFinite(year)) continue;
    const target = (r.target ?? "").trim();
    const out = normalizeOutcome(r.outcome);
    if (!target || !out) continue;
    m.set(`${target}:${year}`, out);
  }
  return m;
}

export function indexPredictions(predObj) {
  // predObj: { predictions: [{year, shadow, groundhogSlug, ...}, ...] }
  const byYear = new Map();
  for (const p of (predObj?.predictions ?? [])) {
    const year = +p.year;
    const slug = p.groundhogSlug;
    if (!Number.isFinite(year) || !slug) continue;
    if (!byYear.has(year)) byYear.set(year, []);
    byYear.get(year).push(p);
  }
  for (const [y, arr] of byYear) {
    arr.sort((a,b)=> String(a.groundhogSlug).localeCompare(String(b.groundhogSlug)));
  }
  return byYear;
}

function addCounts(counts, slug, isCorrect, weight=1) {
  if (!counts.has(slug)) counts.set(slug, { n: 0, k: 0 });
  const c = counts.get(slug);
  c.n += weight;
  c.k += isCorrect ? weight : 0;
}

function trainCounts(predByYear, outcomes, target, yearExclusive, halfLifeYears = 10) {
  const countsRaw = new Map();
  const countsDecayed = new Map();
  const lambda = Math.log(2) / Math.max(1e-9, halfLifeYears);

  for (const [year, preds] of predByYear) {
    if (year >= yearExclusive) continue;
    const actual = outcomes.get(`${target}:${year}`);
    if (!actual) continue;

    const age = (yearExclusive - year);
    const decayW = Math.exp(-lambda * age);

    for (const p of preds) {
      const slug = p.groundhogSlug;
      const predOut = predictionToOutcome(!!p.shadow);
      const correct = (predOut === actual);
      addCounts(countsRaw, slug, correct, 1);
      addCounts(countsDecayed, slug, correct, decayW);
    }
  }
  return { countsRaw, countsDecayed };
}

export function trainWeights(predByYear, outcomes, target, yearExclusive, method, opts) {
  const minObs = opts.minObs ?? 0;
  const alpha = opts.alpha ?? 1.0;
  const gamma = opts.gamma ?? 0.5;
  const [a0, b0] = opts.betaPrior ?? [2, 2];
  const halfLifeYears = opts.halfLifeYears ?? 10;

  const { countsRaw, countsDecayed } = trainCounts(predByYear, outcomes, target, yearExclusive, halfLifeYears);
  const weights = new Map();
  const stats = new Map();

  const usesDecay = (method === "exp_decay" || method === "logit_decay");
  const counts = usesDecay ? countsDecayed : countsRaw;

  for (const [slug, c] of counts) {
    const n = c.n;
    const k = c.k;

    // For filtering, use raw n (count of actual observations), not decayed.
    const nRaw = countsRaw.get(slug)?.n ?? 0;
    if (nRaw < minObs) continue;

    let pHat;
    if (method === "bayes" || method === "exp_decay" || method === "logit" || method === "logit_decay") {
      pHat = (k + a0) / (n + a0 + b0);
    } else { // smoothed accuracy
      pHat = (k + 1) / (n + 2);
    }

    // sample-size boost; gently rewards more evidence.
    const boost = Math.pow(Math.max(1, nRaw), gamma);
    const pClamped = clamp(pHat, 1e-6, 1 - 1e-6);
    let w;
    if (method === "logit" || method === "logit_decay") {
      // Signed weight: contrarian groundhogs (p < 0.5) get negative weight.
      const logit = Math.log(pClamped / (1 - pClamped));
      w = logit * alpha * boost;
    } else {
      w = Math.pow(pClamped, alpha) * boost;
    }

    weights.set(slug, w);
    stats.set(slug, { n: nRaw, k: (countsRaw.get(slug)?.k ?? 0), p: pHat, w });
  }

  // Normalize by sum of absolute weights (works for signed logit weights too).
  const sumAbs = Array.from(weights.values()).reduce((s,x)=>s+Math.abs(x),0) || 1;
  for (const [slug, w] of weights) weights.set(slug, w / sumAbs);

  return { weights, stats };
}

export function ensemblePredictForYear(predByYear, outcomes, target, year, weights) {
  const preds = predByYear.get(year) ?? [];
  const actual = outcomes.get(`${target}:${year}`) ?? "";
  let score = 0;
  let used = 0;

  for (const p of preds) {
    const slug = p.groundhogSlug;
    const w = weights.get(slug);
    if (!w) continue;
    const out = predictionToOutcome(!!p.shadow);
    const vote = (out === ES) ? +1 : -1;
    score += w * vote;
    used++;
  }

  const pred = (score === 0) ? "" : (score > 0 ? ES : LW);
  const correct = actual && pred ? (pred === actual) : null;

  return { year, pred, actual, correct, used, score };
}

export function runBacktest(predByYear, outcomes, target, method, opts) {
  // Candidate years: those with both predictions and outcomes.
  const years = Array.from(predByYear.keys()).sort((a,b)=>a-b)
    .filter(y => outcomes.has(`${target}:${y}`));

  const rows = [];
  for (const y of years) {
    const { weights, stats } = trainWeights(predByYear, outcomes, target, y, method, opts);
    const r = ensemblePredictForYear(predByYear, outcomes, target, y, weights);
    r.weights = weights;
    r.stats = stats;
    rows.push(r);
  }

  // cumulative accuracy (ignoring null)
  let k = 0, n = 0;
  for (const r of rows) {
    if (r.correct === null) continue;
    n += 1;
    if (r.correct) k += 1;
    r.cumAcc = k / n;
    r.cumN = n;
    r.ci = wilsonCI(k, n);
  }

  return rows;
}

export function computeDAE(backtestRows, daeCfg) {
  const minBacktestYears = daeCfg.minBacktestYears ?? 10;
  const maxCIHalfWidth = daeCfg.maxCIHalfWidth ?? 0.10;
  const minGroundhogs = daeCfg.minGroundhogs ?? 8;
  const minObsPerGroundhog = daeCfg.minObsPerGroundhog ?? 5;

  // We define DAE as the first year y such that, looking at results up through y (inclusive):
  // - at least minBacktestYears scored years exist
  // - the Wilson CI half-width for cumulative accuracy is <= maxCIHalfWidth
  // - in year y, at least minGroundhogs had weights and each had >= minObsPerGroundhog historical observations
  for (const r of backtestRows) {
    if (!r.cumN || r.cumN < minBacktestYears) continue;
    if (!r.ci || r.ci.half > maxCIHalfWidth) continue;

    // active groundhogs in this year (have a weight and made a prediction)
    const preds = r.used ?? 0;
    if (preds < minGroundhogs) continue;

    let good = 0;
    for (const [slug, s] of r.stats ?? []) {
      if ((s?.n ?? 0) >= minObsPerGroundhog) good++;
    }
    if (good < minGroundhogs) continue;

    return r.year;
  }
  return null;
}
