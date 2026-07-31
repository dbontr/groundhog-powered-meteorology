import {
  ensemblePredictForYear,
  predictionToOutcome,
  trainWeights
} from "./backtest.js";

export const GOAL_ACCURACY = 0.70;

const DEFAULT_MIN_BACKTEST_GH = 13;
const DEFAULT_CLIMATOLOGY_WINDOW_YEARS = 15;

function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

function logit(p) {
  const q = clamp(p, 1e-6, 1 - 1e-6);
  return Math.log(q / (1 - q));
}

export function majorityVote(preds) {
  let early = 0;
  let late = 0;
  for (const prediction of preds) {
    const outcome = predictionToOutcome(!!prediction.shadow);
    if (outcome === "EARLY_SPRING") early += 1;
    else late += 1;
  }
  const used = early + late;
  if (!used) {
    return { pred: "", probability: 0.5, certainty: 0, margin: 0, used: 0 };
  }
  const margin = (early - late) / used;
  const probability = (margin + 1) / 2;
  const pred = margin > 0 ? "EARLY_SPRING" : margin < 0 ? "LONG_WINTER" : "";
  return { pred, probability, certainty: Math.abs(margin), margin, used };
}

export function signalFromPrediction(result) {
  let probability = Number.isFinite(result?.probability)
    ? result.probability
    : Number.NaN;
  if (!Number.isFinite(probability) && result?.pred) {
    const sign = result.pred === "EARLY_SPRING" ? 1 : -1;
    const certainty = Number.isFinite(result.certainty) ? result.certainty : 0;
    probability = (sign * certainty + 1) / 2;
  }
  if (!Number.isFinite(probability)) probability = 0.5;
  const strength = clamp(2 * probability - 1, -1, 1);
  const signal = strength > 0 ? 1 : strength < 0 ? -1 : 0;
  return { signal, strength };
}

function weightedLogit(value, coefficient) {
  if (!coefficient) return 0;
  return coefficient * logit(value);
}
export function buildFusionWeights(stats, maxN, config) {
  const weights = new Map();
  const denom = Math.log1p(Math.max(1, maxN));

  for (const [slug, stat] of stats) {
    if (stat.n < (config.minObs ?? 0)) continue;
    const evidence = denom ? Math.log1p(stat.n) / denom : 0;
    const stability = Number.isFinite(stat.stability) ? stat.stability : 0.5;
    const trend = Number.isFinite(stat.trend) ? stat.trend : 0;
    const skill = weightedLogit(stat.accBayes, config.wBayes ?? 1)
      + weightedLogit(stat.accDecay, config.wDecay ?? 0)
      + weightedLogit(stat.accWindow, config.wWindow ?? 0)
      + (config.wTrend ?? 0) * trend;

    if (!Number.isFinite(skill)) continue;
    if (!config.contrarian && skill <= 0) continue;

    const evidenceFactor = 1 + (config.wEvidence ?? 0) * evidence;
    const stabilityFactor = Math.max(
      0,
      1 + (config.wStability ?? 0) * (stability - 0.5)
    );
    const boost = Math.pow(Math.max(1, stat.n), config.nBoost ?? 0.5);
    const weight = skill * evidenceFactor * stabilityFactor * boost;
    if (Math.abs(weight) >= 1e-9) weights.set(slug, weight);
  }
  const sumAbs = Array.from(weights.values())
    .reduce((sum, weight) => sum + Math.abs(weight), 0);
  if (!sumAbs) return new Map();
  for (const [slug, weight] of weights) {
    weights.set(slug, weight / sumAbs);
  }
  return weights;
}

function makeBacktestRow(year, result, actual, profileId) {
  return { year, actual, profileId, ...result };
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
  const balancedAccuracy = Number.isFinite(positiveRecall)
    && Number.isFinite(negativeRecall)
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

function climatologyProbability(outcomes, target, year, windowYears) {
  let early = 0;
  let late = 0;
  const prefix = `${target}:`;
  const windowStart = windowYears ? year - windowYears : -Infinity;
  for (const [key, outcome] of outcomes) {
    if (!key.startsWith(prefix)) continue;
    const outcomeYear = Number(key.slice(prefix.length));
    if (!Number.isFinite(outcomeYear)
      || outcomeYear >= year
      || outcomeYear < windowStart) continue;
    if (outcome === "EARLY_SPRING") early += 1;
    else if (outcome === "LONG_WINTER") late += 1;
  }
  return {
    probability: (early + 1) / (early + late + 2),
    observations: early + late
  };
}
export function computeHistoricalClimatologyBacktest(
  outcomes,
  target,
  years,
  opts = {}
) {
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

export const GROUNDHOG_HYBRID = Object.freeze({
  id: "verified-groundhog-wilson",
  reliabilityShare: 1,
  crowdShare: 0,
  reliabilityMethod: "wilson_decay",
  reliabilityOptions: Object.freeze({
    minObs: 8,
    alpha: 1,
    gamma: 0.5,
    betaPrior: [2, 2],
    halfLifeYears: 3,
    windowYears: 20
  })
});

function groundhogHybridPredict(predByYear, outcomes, target, year, opts = {}) {
  const predictions = predByYear.get(year) ?? [];
  if (!predictions.length) {
    return {
      pred: "",
      probability: 0.5,
      certainty: 0,
      margin: 0,
      used: 0,
      weightedUsed: 0,
      usedWeighted: false,
      method: "no-groundhogs"
    };
  }

  const config = {
    ...GROUNDHOG_HYBRID,
    ...opts,
    reliabilityOptions: {
      ...GROUNDHOG_HYBRID.reliabilityOptions,
      ...(opts.reliabilityOptions ?? {})
    }
  };
  const shareTotal = Math.max(
    1e-9,
    config.reliabilityShare + config.crowdShare
  );
  const reliabilityShare = clamp(
    config.reliabilityShare / shareTotal,
    0,
    1
  );
  const { weights } = trainWeights(
    predByYear,
    outcomes,
    target,
    year,
    config.reliabilityMethod,
    config.reliabilityOptions
  );
  const weighted = ensemblePredictForYear(
    predByYear,
    outcomes,
    target,
    year,
    weights
  );
  const hasWeightedSignal = weighted.used > 0 && Number.isFinite(weighted.score);
  const weightedProbability = hasWeightedSignal
    ? clamp((weighted.score + 1) / 2, 0, 1)
    : 0.5;
  const needsCrowd = !hasWeightedSignal || reliabilityShare < 1;
  const crowd = needsCrowd ? majorityVote(predictions) : null;
  const activeReliabilityShare = hasWeightedSignal ? reliabilityShare : 0;
  const activeCrowdShare = 1 - activeReliabilityShare;
  const crowdProbability = crowd?.probability ?? 0.5;
  const probability = activeReliabilityShare * weightedProbability
    + activeCrowdShare * crowdProbability;
  const margin = 2 * probability - 1;
  const pred = margin > 0 ? "EARLY_SPRING" : margin < 0 ? "LONG_WINTER" : "";

  return {
    pred,
    probability,
    certainty: Math.abs(margin),
    margin,
    used: predictions.length,
    weightedUsed: weighted.used,
    usedWeighted: hasWeightedSignal,
    method: hasWeightedSignal ? "verified-groundhog-reliability" : "groundhog-crowd-fallback",
    reliabilityShare: activeReliabilityShare,
    crowdShare: activeCrowdShare,
    weightedProbability,
    crowdProbability
  };
}

export function buildDynamicSuperModel(predByYear, outcomes, target, opts = {}) {
  const minGroundhogs = opts.minGroundhogs ?? DEFAULT_MIN_BACKTEST_GH;
  const hybridOpts = {
    reliabilityShare: opts.reliabilityShare ?? GROUNDHOG_HYBRID.reliabilityShare,
    crowdShare: opts.crowdShare ?? GROUNDHOG_HYBRID.crowdShare,
    reliabilityMethod: opts.reliabilityMethod ?? GROUNDHOG_HYBRID.reliabilityMethod,
    reliabilityOptions: {
      ...GROUNDHOG_HYBRID.reliabilityOptions,
      ...(opts.reliabilityOptions ?? {})
    }
  };
  const allYears = Array.from(predByYear.keys()).sort((a, b) => a - b);
  const scoredYears = allYears.filter((year) => (
    outcomes.has(`${target}:${year}`)
      && (predByYear.get(year)?.length ?? 0) >= minGroundhogs
  ));
  const rows = scoredYears.map((year) => makeBacktestRow(
    year,
    groundhogHybridPredict(predByYear, outcomes, target, year, hybridOpts),
    outcomes.get(`${target}:${year}`),
    GROUNDHOG_HYBRID.id
  ));
  const profileByYear = new Map(
    scoredYears.map((year) => [year, GROUNDHOG_HYBRID.id])
  );
  const currentProfile = { id: GROUNDHOG_HYBRID.id };

  return {
    id: GROUNDHOG_HYBRID.id,
    currentProfile,
    profiles: [currentProfile],
    profileByYear,
    featureCache: { target, allYears, scoredYears },
    outcomes,
    backtest: summarizeBacktestRows(rows),
    hybridOpts,
    selectionMethod: "verified-groundhog-reliability"
  };
}

export function computeDynamicSuperNowcast(predByYear, model) {
  const years = Array.from(predByYear.keys());
  if (!years.length) return null;
  const latestYear = Math.max(...years);
  const predictions = predByYear.get(latestYear) ?? [];
  const result = groundhogHybridPredict(
    predByYear,
    model.outcomes,
    model.featureCache.target,
    latestYear,
    model.hybridOpts
  );

  return {
    latestYear,
    ...result,
    totalPreds: predictions.length,
    profileId: model.id
  };
}
