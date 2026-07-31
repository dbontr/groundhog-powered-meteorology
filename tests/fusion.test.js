import test from "node:test";
import assert from "node:assert/strict";

import {
  buildDynamicSuperModel,
  buildFusionWeights,
  computeDynamicSuperNowcast,
  computeHistoricalClimatologyBacktest,
  GROUNDHOG_HYBRID,
  majorityVote,
  signalFromPrediction
} from "../docs/lib/fusion.js";

const TARGET = "TEST";

function prediction(year, slug, shadow) {
  return { year, groundhogSlug: slug, shadow };
}

test("majority ties carry zero directional confidence", () => {
  const result = majorityVote([
    prediction(2026, "early", false),
    prediction(2026, "late", true)
  ]);

  assert.equal(result.pred, "");
  assert.equal(result.probability, 0.5);
  assert.equal(result.certainty, 0);
});

test("a neutral probability becomes a zero-strength feature", () => {
  assert.deepEqual(
    signalFromPrediction({ pred: "", probability: 0.5, certainty: 0 }),
    { signal: 0, strength: 0 }
  );

  const directional = signalFromPrediction({
    pred: "EARLY_SPRING",
    probability: 0.8,
    certainty: 0.6
  });
  assert.equal(directional.signal, 1);
  assert.ok(Math.abs(directional.strength - 0.6) < 1e-12);
});

test("contrarian mode preserves below-chance negative skill", () => {
  const stats = new Map([["badger", {
    n: 20,
    accBayes: 0.3,
    accDecay: 0.3,
    accWindow: 0.3,
    stability: 1,
    trend: 0
  }]]);

  const baseConfig = {
    minObs: 1,
    nBoost: 0,
    wBayes: 1,
    wDecay: 0,
    wWindow: 0,
    wStability: 0,
    wEvidence: 0,
    wTrend: 0
  };

  const signed = buildFusionWeights(stats, 20, {
    ...baseConfig,
    contrarian: true
  });
  assert.ok(signed.get("badger") < 0);

  const followOnly = buildFusionWeights(stats, 20, {
    ...baseConfig,
    contrarian: false
  });
  assert.equal(followOnly.has("badger"), false);
});

test("historical climatology does not use future outcomes", () => {
  const outcomes = new Map([
    [`${TARGET}:2000`, "EARLY_SPRING"],
    [`${TARGET}:2001`, "LONG_WINTER"],
    [`${TARGET}:2002`, "EARLY_SPRING"]
  ]);
  const first = computeHistoricalClimatologyBacktest(
    outcomes,
    TARGET,
    [2001, 2002]
  );

  outcomes.set(`${TARGET}:2002`, "LONG_WINTER");
  const second = computeHistoricalClimatologyBacktest(
    outcomes,
    TARGET,
    [2001, 2002]
  );

  assert.deepEqual(first.rows[0], second.rows[0]);
  assert.equal(first.rows[0].pred, "EARLY_SPRING");
});

test("groundhog hybrid produces a walk-forward row for every scored year", () => {
  const predByYear = new Map();
  const outcomes = new Map();
  const labels = [false, false, true, false, true, false];

  for (let i = 0; i < labels.length; i++) {
    const year = 2000 + i;
    predByYear.set(year, [
      prediction(year, "alpha", labels[i]),
      prediction(year, "beta", !labels[i])
    ]);
    outcomes.set(
      `${TARGET}:${year}`,
      i % 3 === 1 ? "LONG_WINTER" : "EARLY_SPRING"
    );
  }

  const model = buildDynamicSuperModel(predByYear, outcomes, TARGET, {
    minGroundhogs: 1
  });

  assert.ok(model);
  assert.equal(model.selectionMethod, "fixed-groundhog-hybrid");
  assert.equal(model.id, GROUNDHOG_HYBRID.id);
  assert.equal(model.backtest.rows.length, labels.length);
  assert.equal(model.profileByYear.size, labels.length);
  for (const row of model.backtest.rows) {
    assert.equal(row.profileId, GROUNDHOG_HYBRID.id);
  }
  assert.ok(Number.isFinite(model.backtest.brierScore));
});

test("future labels cannot change earlier walk-forward predictions", () => {
  const predByYear = new Map();
  const outcomes = new Map();
  for (let year = 2000; year <= 2010; year++) {
    predByYear.set(year, [prediction(year, "alpha", year % 2 === 0)]);
    outcomes.set(`${TARGET}:${year}`, year % 4 === 0 ? "LONG_WINTER" : "EARLY_SPRING");
  }

  const first = buildDynamicSuperModel(predByYear, outcomes, TARGET, {
    minGroundhogs: 1
  });
  outcomes.set(`${TARGET}:2010`, "LONG_WINTER");
  const second = buildDynamicSuperModel(predByYear, outcomes, TARGET, {
    minGroundhogs: 1
  });

  const beforeFuture = (model) => model.backtest.rows
    .filter((row) => row.year < 2010)
    .map(({ year, pred, probability, profileId }) => ({ year, pred, probability, profileId }));
  assert.deepEqual(beforeFuture(first), beforeFuture(second));
});

test("nowcast is composed only from reliability and crowd animal votes", () => {
  const predByYear = new Map();
  const outcomes = new Map();
  for (let year = 2000; year <= 2010; year++) {
    predByYear.set(year, [
      prediction(year, "alpha", false),
      prediction(year, "beta", true)
    ]);
    if (year < 2010) outcomes.set(`${TARGET}:${year}`, "EARLY_SPRING");
  }

  const model = buildDynamicSuperModel(predByYear, outcomes, TARGET, {
    minGroundhogs: 1
  });
  const nowcast = computeDynamicSuperNowcast(predByYear, model);

  assert.equal(nowcast.method, "reliability-crowd-hybrid");
  assert.ok(Math.abs(nowcast.reliabilityShare - 0.8) < 1e-12);
  assert.ok(Math.abs(nowcast.crowdShare - 0.2) < 1e-12);
  assert.ok(Math.abs(nowcast.reliabilityShare + nowcast.crowdShare - 1) < 1e-12);
  assert.equal("climatologyProbability" in nowcast, false);
  assert.equal("groundhogWeight" in nowcast, false);
});
