import { predictionToOutcome } from "./backtest.js";

export const GOAL_ACCURACY = 0.70;

export const FUSION_CONFIGS = [
  {
    id: "spark",
    halfLifeYears: 6,
    windowYears: 12,
    minObs: 6,
    nBoost: 0.45,
    priorA: 2,
    priorB: 2,
    wBayes: 1.2,
    wDecay: 1.6,
    wWindow: 0.9,
    wStability: 0.7,
    wEvidence: 0.5,
    wTrend: 0.25,
    contrarian: true
  },
  {
    id: "balanced",
    halfLifeYears: 10,
    windowYears: 18,
    minObs: 8,
    nBoost: 0.55,
    priorA: 3,
    priorB: 3,
    wBayes: 1.4,
    wDecay: 1.2,
    wWindow: 1.0,
    wStability: 0.8,
    wEvidence: 0.6,
    wTrend: 0.2,
    contrarian: true
  },
  {
    id: "stable",
    halfLifeYears: 16,
    windowYears: 26,
    minObs: 10,
    nBoost: 0.6,
    priorA: 4,
    priorB: 4,
    wBayes: 1.6,
    wDecay: 0.9,
    wWindow: 1.2,
    wStability: 1.2,
    wEvidence: 0.7,
    wTrend: 0.1,
    contrarian: false
  },
  {
    id: "recency",
    halfLifeYears: 4,
    windowYears: 9,
    minObs: 5,
    nBoost: 0.4,
    priorA: 2,
    priorB: 2,
    wBayes: 0.9,
    wDecay: 1.8,
    wWindow: 0.8,
    wStability: 0.5,
    wEvidence: 0.4,
    wTrend: 0.35,
    contrarian: true
  },
  {
    id: "legacy",
    halfLifeYears: 22,
    windowYears: 32,
    minObs: 12,
    nBoost: 0.7,
    priorA: 5,
    priorB: 5,
    wBayes: 1.7,
    wDecay: 0.6,
    wWindow: 1.3,
    wStability: 1.0,
    wEvidence: 0.9,
    wTrend: 0.05,
    contrarian: false
  },
  {
    id: "stability",
    halfLifeYears: 12,
    windowYears: 20,
    minObs: 8,
    nBoost: 0.5,
    priorA: 3,
    priorB: 3,
    wBayes: 1.3,
    wDecay: 1.0,
    wWindow: 0.9,
    wStability: 1.4,
    wEvidence: 0.5,
    wTrend: 0.15,
    contrarian: false
  },
  {
    id: "window",
    halfLifeYears: 9,
    windowYears: 10,
    minObs: 5,
    nBoost: 0.45,
    priorA: 2,
    priorB: 2,
    wBayes: 1.0,
    wDecay: 1.0,
    wWindow: 1.6,
    wStability: 0.6,
    wEvidence: 0.4,
    wTrend: 0.2,
    contrarian: true
  },
  {
    id: "momentum",
    halfLifeYears: 5,
    windowYears: 8,
    minObs: 4,
    nBoost: 0.35,
    priorA: 2,
    priorB: 2,
    wBayes: 0.8,
    wDecay: 1.7,
    wWindow: 0.7,
    wStability: 0.5,
    wEvidence: 0.3,
    wTrend: 0.6,
    contrarian: true
  },
  {
    id: "evidence",
    halfLifeYears: 12,
    windowYears: 24,
    minObs: 12,
    nBoost: 0.8,
    priorA: 4,
    priorB: 4,
    wBayes: 1.2,
    wDecay: 0.8,
    wWindow: 0.7,
    wStability: 1.0,
    wEvidence: 1.1,
    wTrend: 0.05,
    contrarian: false
  }
];

export const TUNING_SETS = [
  {
    id: "balanced",
    gate: 0.56,
    topK: 6,
    minModels: 3,
    minUsedRatio: 0.45,
    minConfigYears: 8,
    rankWindowYears: 24,
    rankDecayHalfLife: 12,
    stack: { steps: 260, lr: 0.22, l2: 0.06, minTrain: 12, decayHalfLife: 12, windowYears: 30 },
    blend: { minTrain: 8, windowYears: 26, decayHalfLife: 10, stabilityBoost: 0.8 }
  },
  {
    id: "aggressive",
    gate: 0.60,
    topK: 5,
    minModels: 2,
    minUsedRatio: 0.35,
    minConfigYears: 6,
    rankWindowYears: 18,
    rankDecayHalfLife: 8,
    stack: { steps: 300, lr: 0.26, l2: 0.05, minTrain: 10, decayHalfLife: 8, windowYears: 20 },
    blend: { minTrain: 7, windowYears: 16, decayHalfLife: 6, stabilityBoost: 0.5 }
  },
  {
    id: "steady",
    gate: 0.54,
    topK: 7,
    minModels: 4,
    minUsedRatio: 0.55,
    minConfigYears: 10,
    rankWindowYears: 32,
    rankDecayHalfLife: 16,
    stack: { steps: 240, lr: 0.18, l2: 0.08, minTrain: 14, decayHalfLife: 16, windowYears: 35 },
    blend: { minTrain: 9, windowYears: 30, decayHalfLife: 14, stabilityBoost: 1.1 }
  },
  {
    id: "recency",
    gate: 0.58,
    topK: 4,
    minModels: 2,
    minUsedRatio: 0.35,
    minConfigYears: 6,
    rankWindowYears: 12,
    rankDecayHalfLife: 4,
    stack: { steps: 280, lr: 0.24, l2: 0.05, minTrain: 9, decayHalfLife: 6, windowYears: 12 },
    blend: { minTrain: 7, windowYears: 10, decayHalfLife: 4, stabilityBoost: 0.3 }
  },
  {
    id: "conservative",
    gate: 0.52,
    topK: 8,
    minModels: 5,
    minUsedRatio: 0.6,
    minConfigYears: 12,
    rankWindowYears: 40,
    rankDecayHalfLife: 20,
    stack: { steps: 220, lr: 0.16, l2: 0.1, minTrain: 16, decayHalfLife: 20, windowYears: 40 },
    blend: { minTrain: 10, windowYears: 35, decayHalfLife: 18, stabilityBoost: 1.2 }
  }
];

const DEFAULT_MIN_BACKTEST_GH = 20;

function hasMinGroundhogs(predByYear, year, minCount = DEFAULT_MIN_BACKTEST_GH) {
  const preds = predByYear.get(year) ?? [];
  return preds.length >= minCount;
}


export function majorityVote(preds) {
  let early = 0;
  let late = 0;
  for (const p of preds) {
    const out = predictionToOutcome(!!p.shadow);
    if (out === "EARLY_SPRING") early += 1;
    else late += 1;
  }
  const used = early + late;
  if (!used) {
    return { pred: "", probability: 0.5, certainty: 0, margin: 0, used };
  }
  const margin = (early - late) / used;
  const probability = (margin + 1) / 2;
  const pred = margin > 0 ? "EARLY_SPRING" : margin < 0 ? "LONG_WINTER" : "";
  return { pred, probability, certainty: Math.abs(margin), margin, used };
}

function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

function logit(p) {
  const q = clamp(p, 1e-6, 1 - 1e-6);
  return Math.log(q / (1 - q));
}

function computeFusionStats(predByYear, outcomes, target, yearExclusive, cfg) {
  const years = Array.from(predByYear.keys())
    .filter((y) => y < yearExclusive && outcomes.has(`${target}:${y}`))
    .sort((a, b) => a - b);
  const stats = new Map();
  if (!years.length) return { stats, maxN: 0 };

  const splitYear = years[Math.floor(years.length / 2)];
  const lambda = cfg.halfLifeYears ? Math.log(2) / Math.max(1e-9, cfg.halfLifeYears) : null;
  const windowStart = cfg.windowYears ? yearExclusive - cfg.windowYears : null;
  let maxN = 0;

  for (const year of years) {
    const actual = outcomes.get(`${target}:${year}`);
    const preds = predByYear.get(year) ?? [];
    const age = yearExclusive - year;
    const decayW = lambda ? Math.exp(-lambda * age) : 1;
    const inWindow = windowStart ? year >= windowStart : true;

    for (const p of preds) {
      const slug = p.groundhogSlug;
      if (!slug) continue;
      if (!stats.has(slug)) {
        stats.set(slug, {
          n: 0,
          k: 0,
          nDecay: 0,
          kDecay: 0,
          nWindow: 0,
          kWindow: 0,
          nEarly: 0,
          kEarly: 0,
          nLate: 0,
          kLate: 0
        });
      }
      const s = stats.get(slug);
      const predOut = predictionToOutcome(!!p.shadow);
      const correct = predOut === actual;

      s.n += 1;
      if (correct) s.k += 1;
      s.nDecay += decayW;
      if (correct) s.kDecay += decayW;
      if (inWindow) {
        s.nWindow += 1;
        if (correct) s.kWindow += 1;
      }
      if (year < splitYear) {
        s.nEarly += 1;
        if (correct) s.kEarly += 1;
      } else {
        s.nLate += 1;
        if (correct) s.kLate += 1;
      }

      if (s.n > maxN) maxN = s.n;
    }
  }

  const derived = new Map();
  for (const [slug, s] of stats) {
    if (!s.n) continue;
    const priorA = cfg.priorA ?? 2;
    const priorB = cfg.priorB ?? 2;
    const accRaw = s.k / s.n;
    const accBayes = (s.k + priorA) / (s.n + priorA + priorB);
    const accDecay = s.nDecay ? (s.kDecay + priorA) / (s.nDecay + priorA + priorB) : accBayes;
    const accWindow = s.nWindow ? (s.kWindow + priorA) / (s.nWindow + priorA + priorB) : accBayes;
    const hasBothEras = s.nEarly > 0 && s.nLate > 0;
    const accEarly = s.nEarly ? s.kEarly / s.nEarly : Number.NaN;
    const accLate = s.nLate ? s.kLate / s.nLate : Number.NaN;
    const stability = hasBothEras
      ? clamp(1 - Math.abs(accEarly - accLate), 0, 1)
      : 0.5;
    const trend = hasBothEras ? accLate - accEarly : 0;

    derived.set(slug, {
      n: s.n,
      accRaw,
      accBayes,
      accDecay,
      accWindow,
      stability,
      trend
    });
  }

  return { stats: derived, maxN };
}

export function buildFusionWeights(stats, maxN, cfg) {
  const weights = new Map();
  const denom = Math.log1p(Math.max(1, maxN));

  for (const [slug, s] of stats) {
    if (s.n < (cfg.minObs ?? 0)) continue;

    const evidence = denom ? Math.log1p(s.n) / denom : 0;
    const stability = Number.isFinite(s.stability) ? s.stability : 0.5;
    let skill = (cfg.wBayes ?? 1) * logit(s.accBayes)
      + (cfg.wDecay ?? 0) * logit(s.accDecay)
      + (cfg.wWindow ?? 0) * logit(s.accWindow)
      + (cfg.wTrend ?? 0) * s.trend;

    if (!Number.isFinite(skill)) continue;
    if (!cfg.contrarian && skill <= 0) continue;

    const evidenceFactor = 1 + (cfg.wEvidence ?? 0) * evidence;
    const stabilityFactor = Math.max(
      0,
      1 + (cfg.wStability ?? 0) * (stability - 0.5)
    );
    const boost = Math.pow(Math.max(1, s.n), cfg.nBoost ?? 0.5);
    const w = skill * evidenceFactor * stabilityFactor * boost;
    if (Math.abs(w) < 1e-9) continue;
    weights.set(slug, w);
  }

  const sumAbs = Array.from(weights.values()).reduce((sum, x) => sum + Math.abs(x), 0);
  if (!sumAbs) return new Map();
  for (const [slug, w] of weights) weights.set(slug, w / sumAbs);
  return weights;
}

function predictWithFusionConfig(predByYear, outcomes, target, year, cfg, cache) {
  const key = `${cfg.id}:${year}`;
  if (cache?.has(key)) return cache.get(key);

  const preds = predByYear.get(year) ?? [];
  if (!preds.length) {
    const empty = {
      pred: "",
      probability: 0.5,
      certainty: 0,
      margin: 0,
      used: 0,
      usedWeighted: false
    };
    cache?.set(key, empty);
    return empty;
  }

  const { stats, maxN } = computeFusionStats(predByYear, outcomes, target, year, cfg);
  const weights = buildFusionWeights(stats, maxN, cfg);

  let score = 0;
  let totalAbs = 0;
  let used = 0;

  for (const p of preds) {
    const w = weights.get(p.groundhogSlug);
    if (!w) continue;
    const out = predictionToOutcome(!!p.shadow);
    const vote = (out === "EARLY_SPRING") ? 1 : -1;
    score += w * vote;
    totalAbs += Math.abs(w);
    used += 1;
  }

  let res;
  if (!totalAbs) {
    const fallback = majorityVote(preds);
    res = { ...fallback, usedWeighted: false };
  } else {
    const margin = clamp(score / totalAbs, -1, 1);
    const probability = (margin + 1) / 2;
    const pred = margin > 0 ? "EARLY_SPRING" : margin < 0 ? "LONG_WINTER" : "";
    res = {
      pred,
      probability,
      certainty: Math.abs(margin),
      margin,
      used,
      usedWeighted: true
    };
  }

  cache?.set(key, res);
  return res;
}

export function signalFromPrediction(res) {
  let probability = Number.isFinite(res?.probability) ? res.probability : Number.NaN;
  if (!Number.isFinite(probability) && res?.pred) {
    const sign = res.pred === "EARLY_SPRING" ? 1 : -1;
    const certainty = Number.isFinite(res.certainty) ? res.certainty : 0;
    probability = (sign * certainty + 1) / 2;
  }
  if (!Number.isFinite(probability)) probability = 0.5;
  const strength = clamp(2 * probability - 1, -1, 1);
  const signal = strength > 0 ? 1 : strength < 0 ? -1 : 0;
  return { signal, strength };
}

function buildFusionFeatureCache(predByYear, outcomes, target, configs, minGroundhogs) {
  const allYears = Array.from(predByYear.keys()).sort((a, b) => a - b);
  const scoredYears = allYears.filter((y) => outcomes.has(`${target}:${y}`) && hasMinGroundhogs(predByYear, y, minGroundhogs));
  const resultsByYear = new Map();
  const predCache = new Map();

  for (const y of allYears) {
    const results = configs.map((cfg) => predictWithFusionConfig(predByYear, outcomes, target, y, cfg, predCache));
    resultsByYear.set(y, results);
  }

  const labelsByYear = new Map();
  for (const y of scoredYears) {
    const out = outcomes.get(`${target}:${y}`);
    labelsByYear.set(y, out === "EARLY_SPRING" ? 1 : 0);
  }

  return { target, configs, allYears, scoredYears, resultsByYear, labelsByYear };
}

function configPerformance(featureCache, configIndex, trainYears, opts = {}) {
  let k = 0;
  let n = 0;
  let kEarly = 0;
  let nEarly = 0;
  let kLate = 0;
  let nLate = 0;
  const currentYear = opts.currentYear ?? Math.max(...trainYears);
  const lambda = opts.decayHalfLife ? Math.log(2) / Math.max(1e-9, opts.decayHalfLife) : null;
  const splitYear = trainYears[Math.floor(trainYears.length / 2)];

  for (const y of trainYears) {
    const res = featureCache.resultsByYear.get(y)?.[configIndex];
    if (!res?.pred) continue;
    const actual = featureCache.labelsByYear.get(y) === 1 ? "EARLY_SPRING" : "LONG_WINTER";
    const correct = res.pred === actual;
    const age = currentYear - y;
    const weight = lambda ? Math.exp(-lambda * age) : 1;

    n += weight;
    if (correct) k += weight;

    if (y < splitYear) {
      nEarly += weight;
      if (correct) kEarly += weight;
    } else {
      nLate += weight;
      if (correct) kLate += weight;
    }
  }

  const acc = n ? k / n : Number.NaN;
  const accEarly = nEarly ? kEarly / nEarly : acc;
  const accLate = nLate ? kLate / nLate : acc;
  const stability = Number.isFinite(accEarly) && Number.isFinite(accLate)
    ? clamp(1 - Math.abs(accEarly - accLate), 0, 1)
    : 0.5;

  return { k, n, acc, stability };
}

function selectTopConfigIndexes(featureCache, year, cfg) {
  const windowStart = cfg.rankWindowYears ? year - cfg.rankWindowYears : null;
  const trainYears = featureCache.scoredYears.filter((y) => y < year && (!windowStart || y >= windowStart));
  if (!trainYears.length) return featureCache.configs.map((_, idx) => idx);

  const scored = featureCache.configs.map((c, idx) => {
    const perf = configPerformance(featureCache, idx, trainYears, {
      currentYear: year,
      decayHalfLife: cfg.rankDecayHalfLife
    });
    if (!Number.isFinite(perf.acc) || perf.n < (cfg.minConfigYears ?? 6)) return null;
    const p = (perf.k + 2) / (perf.n + 4);
    return { idx, score: p, n: perf.n };
  }).filter(Boolean);

  if (!scored.length) return featureCache.configs.map((_, idx) => idx);
  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return b.n - a.n;
  });

  const topK = Math.max(1, Math.min(cfg.topK ?? scored.length, scored.length));
  return scored.slice(0, topK).map((s) => s.idx);
}

function buildFeatureVector(featureCache, year, configIdxs) {
  const results = featureCache.resultsByYear.get(year) ?? [];
  const strengths = [];
  let used = 0;
  let sumSignal = 0;
  let sumStrength = 0;
  let sumAbsStrength = 0;

  for (const idx of configIdxs) {
    const res = results[idx];
    const { signal, strength } = signalFromPrediction(res);
    if (signal !== 0) {
      used += 1;
      strengths.push(strength);
    }
    sumSignal += signal;
    sumStrength += strength;
    sumAbsStrength += Math.abs(strength);
  }

  const usedRatio = configIdxs.length ? used / configIdxs.length : 0;
  const meanSignal = used ? sumSignal / used : 0;
  const meanStrength = used ? sumStrength / used : 0;
  const meanAbsStrength = used ? sumAbsStrength / used : 0;
  const consensus = used ? Math.abs(sumSignal) / used : 0;
  const disagreement = 1 - consensus;
  const variance = strengths.length
    ? strengths.reduce((sum, x) => sum + ((x - meanStrength) ** 2), 0) / strengths.length
    : 0;
  const strengthSpread = Math.sqrt(variance);
  const feats = [
    meanSignal,
    meanStrength,
    meanAbsStrength,
    consensus,
    disagreement,
    usedRatio,
    strengthSpread
  ];

  return { feats, used };
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

function stackedFusionPredict(featureCache, year, configIdxs, opts = {}) {
  const windowStart = opts.windowYears ? year - opts.windowYears : null;
  const trainYears = featureCache.scoredYears.filter((y) => y < year && (!windowStart || y >= windowStart));
  if (trainYears.length < (opts.minTrain ?? 12)) {
    return { pred: "", probability: 0.5, certainty: 0, margin: 0, used: 0 };
  }

  const X = [];
  const y = [];
  const sampleWeights = [];
  const lambda = opts.decayHalfLife ? Math.log(2) / Math.max(1e-9, opts.decayHalfLife) : null;

  for (const trainYear of trainYears) {
    const { feats, used } = buildFeatureVector(featureCache, trainYear, configIdxs);
    if (!used) continue;
    X.push(feats);
    y.push(featureCache.labelsByYear.get(trainYear));
    if (lambda) sampleWeights.push(Math.exp(-lambda * (year - trainYear)));
  }

  if (X.length < (opts.minTrain ?? 12)) {
    return { pred: "", probability: 0.5, certainty: 0, margin: 0, used: 0 };
  }

  const model = trainLogistic(X, y, { ...opts, sampleWeights: lambda ? sampleWeights : null });
  if (!model) return { pred: "", probability: 0.5, certainty: 0, margin: 0, used: 0 };

  const { feats, used } = buildFeatureVector(featureCache, year, configIdxs);
  if (!used) return { pred: "", probability: 0.5, certainty: 0, margin: 0, used: 0 };

  let z = model.b;
  for (let j = 0; j < model.w.length; j++) z += model.w[j] * feats[j];
  z = clamp(z, -35, 35);
  const probability = 1 / (1 + Math.exp(-z));
  const margin = 2 * probability - 1;
  const pred = margin > 0 ? "EARLY_SPRING" : margin < 0 ? "LONG_WINTER" : "";
  return { pred, probability, certainty: Math.abs(margin), margin, used };
}

function weightedBlendPredict(featureCache, year, configIdxs, opts = {}) {
  const windowStart = opts.windowYears ? year - opts.windowYears : null;
  const trainYears = featureCache.scoredYears.filter((y) => y < year && (!windowStart || y >= windowStart));
  if (trainYears.length < (opts.minTrain ?? 8)) {
    return { pred: "", probability: 0.5, certainty: 0, margin: 0, used: 0 };
  }

  let score = 0;
  let totalAbs = 0;
  let used = 0;

  for (const idx of configIdxs) {
    const perf = configPerformance(featureCache, idx, trainYears, {
      currentYear: year,
      decayHalfLife: opts.decayHalfLife
    });
    if (perf.n < (opts.minConfigYears ?? 6)) continue;
    const p = (perf.k + 2) / (perf.n + 4);
    const stabilityFactor = 1 + (opts.stabilityBoost ?? 0) * (perf.stability - 0.5);
    const weight = logit(p) * (1 + Math.log1p(perf.n) / 3) * stabilityFactor;
    const res = featureCache.resultsByYear.get(year)?.[idx];
    const { signal, strength } = signalFromPrediction(res);
    if (!signal) continue;
    score += weight * strength;
    totalAbs += Math.abs(weight);
    used += 1;
  }

  if (!totalAbs) {
    return { pred: "", probability: 0.5, certainty: 0, margin: 0, used };
  }
  const margin = clamp(score / totalAbs, -1, 1);
  const probability = (margin + 1) / 2;
  const pred = margin > 0 ? "EARLY_SPRING" : margin < 0 ? "LONG_WINTER" : "";
  return { pred, probability, certainty: Math.abs(margin), margin, used };
}

function dynamicSuperPredict(predByYear, year, model) {
  const configIdxs = selectTopConfigIndexes(model.featureCache, year, model);
  let res = stackedFusionPredict(model.featureCache, year, configIdxs, model.stack);
  let method = "stacked";
  const usedRatio = configIdxs.length ? res.used / configIdxs.length : 0;

  if (!res.pred || res.certainty < model.gate || res.used < model.minModels || usedRatio < (model.minUsedRatio ?? 0)) {
    const blended = weightedBlendPredict(model.featureCache, year, configIdxs, {
      ...model.blend,
      minConfigYears: model.minConfigYears
    });
    if (blended.pred) {
      res = blended;
      method = "blend";
    }
  }

  if (!res.pred) {
    res = majorityVote(predByYear.get(year) ?? []);
    method = "majority";
  }

  return { ...res, method, usedWeighted: method !== "majority" };
}

const DEFAULT_CLIMATOLOGY_WINDOW_YEARS = 15;
const DEFAULT_MIN_BLEND_YEARS = 8;
const DEFAULT_BLEND_SHRINKAGE = 10;

function climatologyProbability(outcomes, target, year, windowYears) {
  let early = 0;
  let late = 0;
  const prefix = `${target}:`;
  const windowStart = windowYears ? year - windowYears : -Infinity;
  for (const [key, outcome] of outcomes) {
    if (!key.startsWith(prefix)) continue;
    const outcomeYear = Number(key.slice(prefix.length));
    if (!Number.isFinite(outcomeYear) || outcomeYear >= year || outcomeYear < windowStart) continue;
    if (outcome === "EARLY_SPRING") early += 1;
    else if (outcome === "LONG_WINTER") late += 1;
  }
  return { probability: (early + 1) / (early + late + 2), observations: early + late };
}

function rawPredictionForProfile(predByYear, profile, year, cache) {
  const key = `${profile.id}:${year}`;
  if (!cache.has(key)) cache.set(key, dynamicSuperPredict(predByYear, year, profile));
  return cache.get(key);
}

function estimateGroundhogWeight(predByYear, outcomes, profile, featureCache, year, rawCache, opts) {
  const historyYears = featureCache.scoredYears.filter((y) => y < year);
  if (historyYears.length < opts.minBlendYears) return 0;
  let climateLoss = 0;
  let groundhogLoss = 0;

  for (const historyYear of historyYears) {
    const label = outcomes.get(`${featureCache.target}:${historyYear}`) === "EARLY_SPRING" ? 1 : 0;
    const climate = climatologyProbability(
      outcomes,
      featureCache.target,
      historyYear,
      opts.climatologyWindowYears
    ).probability;
    const raw = rawPredictionForProfile(predByYear, profile, historyYear, rawCache);
    const groundhog = Number.isFinite(raw.probability) ? raw.probability : 0.5;
    climateLoss += (climate - label) ** 2;
    groundhogLoss += (groundhog - label) ** 2;
  }

  climateLoss /= historyYears.length;
  groundhogLoss /= historyYears.length;
  if (groundhogLoss >= climateLoss || climateLoss <= 0) return 0;
  const relativeSkill = (climateLoss - groundhogLoss) / climateLoss;
  const evidence = historyYears.length / (historyYears.length + opts.blendShrinkage);
  return clamp(relativeSkill * evidence, 0, 1);
}

function guardedPredictionForProfile(
  predByYear,
  outcomes,
  profile,
  featureCache,
  year,
  rawCache,
  guardedCache,
  opts
) {
  const key = `${profile.id}:${year}`;
  if (guardedCache.has(key)) return guardedCache.get(key);
  const raw = rawPredictionForProfile(predByYear, profile, year, rawCache);
  const climate = climatologyProbability(
    outcomes,
    featureCache.target,
    year,
    opts.climatologyWindowYears
  );
  const groundhogWeight = estimateGroundhogWeight(
    predByYear,
    outcomes,
    profile,
    featureCache,
    year,
    rawCache,
    opts
  );
  const rawProbability = Number.isFinite(raw.probability) ? raw.probability : 0.5;
  const probability = (1 - groundhogWeight) * climate.probability
    + groundhogWeight * rawProbability;
  const margin = 2 * probability - 1;
  const pred = margin > 0 ? "EARLY_SPRING" : margin < 0 ? "LONG_WINTER" : "";
  const result = {
    ...raw,
    pred,
    probability,
    certainty: Math.abs(margin),
    margin,
    method: groundhogWeight > 0 ? `${raw.method}+climatology` : "climatology-guard",
    usedWeighted: groundhogWeight > 0 && raw.usedWeighted,
    rawProbability,
    climatologyProbability: climate.probability,
    climatologyObservations: climate.observations,
    groundhogWeight
  };
  guardedCache.set(key, result);
  return result;
}

function makeBacktestRow(year, res, actual, profileId) {
  return { year, actual, profileId, ...res };
}

export function summarizeBacktestRows(rows) {
  let correct = 0;
  let predicted = 0;
  let total = 0;
  let positives = 0;
  let negatives = 0;
  let truePositives = 0;
  let trueNegatives = 0;
  let brier = 0;
  let logLoss = 0;

  for (const row of rows) {
    if (row.actual !== "EARLY_SPRING" && row.actual !== "LONG_WINTER") continue;
    const label = row.actual === "EARLY_SPRING" ? 1 : 0;
    const probability = clamp(
      Number.isFinite(row.probability) ? row.probability : 0.5,
      0,
      1
    );
    const logProbability = clamp(probability, 1e-6, 1 - 1e-6);
    total += 1;
    brier += (probability - label) ** 2;
    logLoss += -(label * Math.log(logProbability)
      + (1 - label) * Math.log(1 - logProbability));
    if (label) positives += 1;
    else negatives += 1;

    if (!row.pred) continue;
    predicted += 1;
    if (row.pred === row.actual) correct += 1;
    if (label && row.pred === "EARLY_SPRING") truePositives += 1;
    if (!label && row.pred === "LONG_WINTER") trueNegatives += 1;
  }

  const positiveRecall = positives ? truePositives / positives : Number.NaN;
  const negativeRecall = negatives ? trueNegatives / negatives : Number.NaN;
  const balancedAccuracy = Number.isFinite(positiveRecall) && Number.isFinite(negativeRecall)
    ? (positiveRecall + negativeRecall) / 2
    : Number.NaN;

  return {
    rows,
    accuracy: predicted ? correct / predicted : Number.NaN,
    balancedAccuracy,
    brierScore: total ? brier / total : Number.NaN,
    logLoss: total ? logLoss / total : Number.NaN,
    coverage: total ? predicted / total : 0,
    backtestN: total,
    predictedN: predicted,
    lastYear: rows.length ? rows[rows.length - 1].year : null,
    confusion: { positives, negatives, truePositives, trueNegatives }
  };
}

const MIN_PROFILE_SELECTION_YEARS = 6;

function buildProfileModels(featureCache, tuningSets) {
  return tuningSets.map((tuning) => ({
    ...tuning,
    target: featureCache.target,
    configs: featureCache.configs,
    featureCache
  }));
}

function selectProfileForYear(
  predByYear,
  outcomes,
  profiles,
  featureCache,
  year,
  rawCache,
  guardedCache,
  guardOpts,
  minYears
) {
  const historyYears = featureCache.scoredYears.filter((y) => y < year);
  if (historyYears.length < minYears) return profiles[0];

  let best = null;
  for (const profile of profiles) {
    const rows = historyYears.map((historyYear) => makeBacktestRow(
      historyYear,
      guardedPredictionForProfile(
        predByYear,
        outcomes,
        profile,
        featureCache,
        historyYear,
        rawCache,
        guardedCache,
        guardOpts
      ),
      outcomes.get(`${featureCache.target}:${historyYear}`),
      profile.id
    ));
    const metrics = summarizeBacktestRows(rows);
    const balanced = Number.isFinite(metrics.balancedAccuracy)
      ? metrics.balancedAccuracy
      : -Infinity;
    const accuracy = Number.isFinite(metrics.accuracy) ? metrics.accuracy : -Infinity;
    const candidate = { profile, metrics, balanced, accuracy };

    if (!best || metrics.brierScore < best.metrics.brierScore - 1e-12) {
      best = candidate;
      continue;
    }
    if (Math.abs(metrics.brierScore - best.metrics.brierScore) <= 1e-12) {
      if (balanced > best.balanced || (balanced === best.balanced && accuracy > best.accuracy)) {
        best = candidate;
      }
    }
  }

  return best?.profile ?? profiles[0];
}

export function computeHistoricalClimatologyBacktest(outcomes, target, years, opts = {}) {
  const windowYears = opts.windowYears ?? DEFAULT_CLIMATOLOGY_WINDOW_YEARS;
  const rows = [...years].sort((a, b) => a - b).map((year) => {
    const climate = climatologyProbability(outcomes, target, year, windowYears);
    const probability = climate.probability;
    const margin = 2 * probability - 1;
    const pred = margin > 0 ? "EARLY_SPRING" : margin < 0 ? "LONG_WINTER" : "";
    return makeBacktestRow(year, {
      pred,
      probability,
      certainty: Math.abs(margin),
      margin,
      used: climate.observations,
      method: "climatology",
      usedWeighted: false
    }, outcomes.get(`${target}:${year}`), "climatology");
  });

  return summarizeBacktestRows(rows);
}

export function buildDynamicSuperModel(predByYear, outcomes, target, opts = {}) {
  const configs = opts.configs ?? FUSION_CONFIGS;
  const tuningSets = opts.tuningSets ?? TUNING_SETS;
  if (!configs.length || !tuningSets.length) return null;
  const minGroundhogs = opts.minGroundhogs ?? DEFAULT_MIN_BACKTEST_GH;
  const minSelectionYears = opts.minSelectionYears ?? MIN_PROFILE_SELECTION_YEARS;
  const guardOpts = {
    climatologyWindowYears: opts.climatologyWindowYears ?? DEFAULT_CLIMATOLOGY_WINDOW_YEARS,
    minBlendYears: opts.minBlendYears ?? DEFAULT_MIN_BLEND_YEARS,
    blendShrinkage: opts.blendShrinkage ?? DEFAULT_BLEND_SHRINKAGE
  };
  const featureCache = buildFusionFeatureCache(predByYear, outcomes, target, configs, minGroundhogs);
  const profiles = buildProfileModels(featureCache, tuningSets);
  const rawPredictionCache = new Map();
  const guardedPredictionCache = new Map();
  const profileByYear = new Map();
  const rows = [];

  for (const year of featureCache.scoredYears) {
    const profile = selectProfileForYear(
      predByYear,
      outcomes,
      profiles,
      featureCache,
      year,
      rawPredictionCache,
      guardedPredictionCache,
      guardOpts,
      minSelectionYears
    );
    profileByYear.set(year, profile.id);
    rows.push(makeBacktestRow(
      year,
      guardedPredictionForProfile(
        predByYear,
        outcomes,
        profile,
        featureCache,
        year,
        rawPredictionCache,
        guardedPredictionCache,
        guardOpts
      ),
      outcomes.get(`${target}:${year}`),
      profile.id
    ));
  }

  const backtest = summarizeBacktestRows(rows);
  const latestYear = featureCache.allYears.length
    ? featureCache.allYears[featureCache.allYears.length - 1]
    : null;
  const currentProfile = latestYear === null
    ? profiles[0]
    : selectProfileForYear(
      predByYear,
      outcomes,
      profiles,
      featureCache,
      latestYear,
      rawPredictionCache,
      guardedPredictionCache,
      guardOpts,
      minSelectionYears
    );

  return {
    ...currentProfile,
    currentProfile,
    profiles,
    profileByYear,
    featureCache,
    outcomes,
    backtest,
    guardOpts,
    selectionMethod: "nested-walk-forward",
    minSelectionYears
  };
}

export function computeDynamicSuperNowcast(predByYear, model) {
  const years = Array.from(predByYear.keys());
  if (!years.length) return null;
  const latestYear = Math.max(...years);
  const preds = predByYear.get(latestYear) ?? [];
  const profile = model.currentProfile ?? model;
  const res = model.outcomes && model.guardOpts
    ? guardedPredictionForProfile(
      predByYear,
      model.outcomes,
      profile,
      model.featureCache,
      latestYear,
      new Map(),
      new Map(),
      model.guardOpts
    )
    : dynamicSuperPredict(predByYear, latestYear, profile);
  return {
    latestYear,
    pred: res.pred,
    probability: res.probability,
    certainty: res.certainty,
    margin: res.margin,
    used: preds.length,
    totalPreds: preds.length,
    usedWeighted: res.usedWeighted,
    method: res.method,
    profileId: profile.id,
    groundhogWeight: res.groundhogWeight ?? 1,
    rawProbability: res.rawProbability ?? res.probability,
    climatologyProbability: res.climatologyProbability ?? Number.NaN
  };
}
