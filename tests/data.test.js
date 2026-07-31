import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

async function loadJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

test("cached forecast data contains verified groundhogs only", async () => {
  const [directory, predictionData] = await Promise.all([
    loadJson("docs/data/groundhogs.json"),
    loadJson("docs/data/predictions.json")
  ]);

  assert.equal(directory.groundhogsOnly, true);
  assert.equal(predictionData.groundhogsOnly, true);
  assert.equal(predictionData.excludesMissingPredictions, true);

  const groundhogs = directory.groundhogs ?? [];
  assert.ok(groundhogs.length > 0);
  assert.ok(groundhogs.every(g => g.isGroundhog === true));
  const allowedSlugs = new Set(groundhogs.map(g => g.slug));

  const predictions = predictionData.predictions ?? [];
  assert.ok(predictions.length > 0);
  assert.ok(predictions.every(p => allowedSlugs.has(p.groundhogSlug)));
  assert.ok(predictions.every(p => typeof p.shadow === "boolean"));
});

test("known missing records do not become no-shadow votes", async () => {
  const predictionData = await loadJson("docs/data/predictions.json");
  const keys = new Set((predictionData.predictions ?? []).map(
    p => `${p.groundhogSlug}:${p.year}`
  ));

  assert.equal(keys.has("punxsutawney-phil:1889"), false);
  assert.equal(keys.has("punxsutawney-phil:1891"), false);
  assert.equal(keys.has("punxsutawney-phil:1900"), true);
});
