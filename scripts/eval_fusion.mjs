import { promises as fs } from "node:fs";
import { performance } from "node:perf_hooks";

import { indexOutcomes, indexPredictions } from "../docs/lib/backtest.js";
import {
  buildDynamicSuperModel,
  computeDynamicSuperNowcast,
  computeHistoricalClimatologyBacktest
} from "../docs/lib/fusion.js";
import { parseCSV } from "../docs/lib/stats.js";

const TARGET = "US_CONUS_FEBMAR_MEAN_ANOM";

function pct(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "n/a";
}

const [predictionText, outcomeText, groundhogText] = await Promise.all([
  fs.readFile("docs/data/predictions.json", "utf8"),
  fs.readFile("docs/data/outcomes.csv", "utf8"),
  fs.readFile("docs/data/groundhogs.json", "utf8")
]);
const predictionData = JSON.parse(predictionText);
const groundhogData = JSON.parse(groundhogText);
if (!predictionData.groundhogsOnly || !predictionData.excludesMissingPredictions) {
  throw new Error("Prediction data is not marked as cleaned groundhog-only data.");
}
if ((groundhogData.groundhogs ?? []).some(g => g.isGroundhog !== true)) {
  throw new Error("Groundhog directory contains a non-groundhog entry.");
}
const verifiedSlugs = new Set(
  (groundhogData.groundhogs ?? []).map(g => g.slug).filter(Boolean)
);
const predByYear = indexPredictions(predictionData, verifiedSlugs);
const outcomes = indexOutcomes(parseCSV(outcomeText));
const modelStart = performance.now();
const model = buildDynamicSuperModel(predByYear, outcomes, TARGET);

if (!model) throw new Error("Dynamic model could not be evaluated.");

const baseline = computeHistoricalClimatologyBacktest(
  outcomes,
  TARGET,
  model.featureCache.scoredYears,
  { windowYears: 15 }
);
const nowcast = computeDynamicSuperNowcast(predByYear, model);
const modelMs = performance.now() - modelStart;

console.log(`Target: ${TARGET}`);
console.log(`Selection: ${model.selectionMethod}`);
console.log(`Backtest years: ${model.backtest.backtestN}`);
console.log(`Accuracy: ${pct(model.backtest.accuracy)}`);
console.log(`Balanced accuracy: ${pct(model.backtest.balancedAccuracy)}`);
console.log(`Brier score: ${model.backtest.brierScore.toFixed(3)}`);
console.log(`Log loss: ${model.backtest.logLoss.toFixed(3)}`);
console.log(`Climatology accuracy: ${pct(baseline.accuracy)}`);
console.log(`Climatology Brier score: ${baseline.brierScore.toFixed(3)}`);
console.log(`Verified groundhogs: ${(groundhogData.groundhogs ?? []).length}`);
console.log(`Model runtime: ${modelMs.toFixed(2)} ms`);
if (nowcast) {
  console.log(`Forecast ${nowcast.latestYear}: ${nowcast.pred || "NO CALL"}`);
  console.log(`Early-spring probability: ${pct(nowcast.probability)}`);
  console.log(`Verified-groundhog share: ${pct(nowcast.reliabilityShare)}`);
  console.log(`Reliability histories: ${nowcast.weightedUsed}/${nowcast.totalPreds}`);
  console.log(`Profile: ${nowcast.profileId}`);
}
