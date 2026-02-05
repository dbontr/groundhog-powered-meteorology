import { parseCSV } from "./lib/stats.js";
import { indexOutcomes, indexPredictions, predictionToOutcome } from "./lib/backtest.js";
import { GOAL_ACCURACY, buildDynamicSuperModel, computeDynamicSuperNowcast } from "./lib/fusion.js";

const $ = (id) => document.getElementById(id);

const TARGET_BASE = "US_CONUS_FEBMAR_MEAN_ANOM";
const TARGET_MARCH = "US_CONUS_MAR_ANOM";
const MIN_OBS = 20;
const MIN_BACKTEST_GH = 20;
const LEADERBOARD_DEFAULT_MIN_OBS = MIN_OBS;

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

function hasMinGroundhogs(predByYear, year, minCount = MIN_BACKTEST_GH) {
  const preds = predByYear.get(year) ?? [];
  return preds.length >= minCount;
}

function computeBaselineAccuracy(outcomes, target, years) {
  let early = 0;
  let late = 0;
  for (const y of years) {
    const actual = outcomes.get(`${target}:${y}`);
    if (actual === "EARLY_SPRING") early += 1;
    else if (actual === "LONG_WINTER") late += 1;
  }
  const total = early + late;
  if (!total) return { accuracy: Number.NaN, majorityOutcome: "", total: 0 };
  const majorityOutcome = early >= late ? "EARLY_SPRING" : "LONG_WINTER";
  return { accuracy: Math.max(early, late) / total, majorityOutcome, total };
}

function computeLeaderboard(predByYear, outcomes, groundhogDir, minObs = MIN_OBS) {
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
  })).filter(r => r.n >= minObs);

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
          <th>Observations</th>
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
        <th>Observations</th>
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

    const scoredYears = Array.from(predByYear.keys())
      .filter((y) => outcomes.has(`${TARGET_BASE}:${y}`) && hasMinGroundhogs(predByYear, y));
    const minYear = scoredYears.length ? Math.min(...scoredYears) : null;
    const maxYear = scoredYears.length ? Math.max(...scoredYears) : null;
    const leaderboardButton = $("toggleNewbies");
    let allowNewbies = false;
    const buildLeaderboardMeta = (minObs) => {
      const obsText = minObs <= 1
        ? "Min observations: none (newbies included)."
        : `Min observations: ${minObs}.`;
      const yearText = scoredYears.length ? ` Scored years: ${minYear}–${maxYear}.` : "";
      return `Computed from groundhog-day.com predictions vs NOAA Climate-at-a-Glance outcomes (CONUS Feb+Mar). ${obsText} Min groundhogs/year: ${MIN_BACKTEST_GH}.${yearText}`;
    };
    const updateLeaderboard = () => {
      const minObs = allowNewbies ? 1 : LEADERBOARD_DEFAULT_MIN_OBS;
      renderLeaderboard(computeLeaderboard(predByYear, outcomes, groundhogDir, minObs));
      $("leaderboardMeta").textContent = buildLeaderboardMeta(minObs);
      if (leaderboardButton) {
        leaderboardButton.textContent = allowNewbies ? "Hide Newbies" : "Allow Newbies";
        leaderboardButton.setAttribute("aria-pressed", String(allowNewbies));
        leaderboardButton.dataset.active = allowNewbies ? "true" : "false";
      }
    };
    updateLeaderboard();
    if (leaderboardButton) {
      leaderboardButton.addEventListener("click", () => {
        allowNewbies = !allowNewbies;
        updateLeaderboard();
      });
    }

    const baseline = computeBaselineAccuracy(outcomes, TARGET_BASE, scoredYears);
    $("stationAccuracy").textContent = fmtPct(baseline.accuracy);
    const stationDetail = $("stationDetail");
    if (stationDetail) {
      stationDetail.textContent = baseline.majorityOutcome
        ? `Climatology proxy: always predict ${outcomeLabel(baseline.majorityOutcome)}.`
        : "";
    }

    const isSample = String(predObj.updatedAt || "").includes("SAMPLE");
    if (isSample) {
      setStatus("Sample data loaded — run python3 scripts/update_predictions.py for the full leaderboard.");
    }

    const model = buildDynamicSuperModel(predByYear, outcomes, TARGET_BASE);
    if (!model) {
      setStatus("Backtest unavailable — dynamic super algorithm could not be evaluated.");
      return;
    }

    const nowcast = computeDynamicSuperNowcast(predByYear, model);
    if (!nowcast || !nowcast.pred) {
      setStatus("No prediction data available.");
      return;
    }

    const indicator = $("indicator");
    indicator.textContent = outcomeLabel(nowcast.pred);
    document.body.dataset.outcome = nowcast.pred;

    const displayCertainty = nowcast.certainty;
    const calibratedCertainty = calibrateCertainty(nowcast.certainty, model.backtest.accuracy, model.backtest.backtestN);
    $("certainty").textContent = fmtPct(displayCertainty, 1);
    $("algoAccuracy").textContent = fmtPct(model.backtest.accuracy);
    $("callYear").textContent = `Forecast for ${nowcast.latestYear}`;
    $("predictionYear").textContent = `${nowcast.latestYear}`;
    if (nowcast.totalPreds) {
      $("voterCount").textContent = `${nowcast.totalPreds} total`;
    } else {
      $("voterCount").textContent = `${nowcast.used}`;
    }

    let metaText = "Dynamic super algorithm fuses multi-signal weights, recency, stability, and stacked learning.";
    if (nowcast.method === "stacked") {
      metaText = "Dynamic super algorithm used stacked fusion across the top-performing signal packs this year.";
    } else if (nowcast.method === "blend") {
      metaText = "Dynamic super algorithm used a weighted blend fallback (stacked confidence below gate).";
    } else if (nowcast.method === "majority") {
      metaText = "Dynamic super algorithm fell back to simple majority vote for this year.";
    }

    metaText += ` Goal accuracy: ${fmtPct(GOAL_ACCURACY, 0)}.`;
    metaText += ` Backtest: ${fmtPct(model.backtest.accuracy)} across ${model.backtest.backtestN} years.`;
    if (Number.isFinite(model.backtest.accuracy)) {
      metaText += model.backtest.accuracy >= GOAL_ACCURACY
        ? " Goal reached in backtest."
        : " Goal not yet reached in backtest.";
    }
    if (model.id) {
      metaText += ` Fusion profile: ${model.id}.`;
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
