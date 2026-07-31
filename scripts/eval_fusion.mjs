import { promises as fs } from "node:fs";

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

const [predictionText, outcomeText] = await Promise.all([
  fs.readFile("docs/data/predictions.json", "utf8"),
  fs.readFile("docs/data/outcomes.csv", "utf8")
]);
const predByYear = indexPredictions(JSON.parse(predictionText));
const outcomes = indexOutcomes(parseCSV(outcomeText));
const model = buildDynamicSuperModel(predByYear, outcomes, TARGET);

if (!model) throw new Error("Dynamic model could not be evaluated.");

const baseline = computeHistoricalClimatologyBacktest(
  outcomes,
  TARGET,
  model.featureCache.scoredYears,
  { windowYears: model.guardOpts.climatologyWindowYears }
);
const nowcast = computeDynamicSuperNowcast(predByYear, model);

console.log(`Target: ${TARGET}`);
console.log(`Selection: ${model.selectionMethod}`);
console.log(`Backtest years: ${model.backtest.backtestN}`);
console.log(`Accuracy: ${pct(model.backtest.accuracy)}`);
console.log(`Balanced accuracy: ${pct(model.backtest.balancedAccuracy)}`);
console.log(`Brier score: ${model.backtest.brierScore.toFixed(3)}`);
console.log(`Log loss: ${model.backtest.logLoss.toFixed(3)}`);
console.log(`Climatology accuracy: ${pct(baseline.accuracy)}`);
console.log(`Climatology Brier score: ${baseline.brierScore.toFixed(3)}`);
if (nowcast) {
  console.log(`Forecast ${nowcast.latestYear}: ${nowcast.pred || "NO CALL"}`);
  console.log(`Early-spring probability: ${pct(nowcast.probability)}`);
  console.log(`Groundhog contribution: ${pct(nowcast.groundhogWeight)}`);
  console.log(`Profile: ${nowcast.profileId}`);
}
