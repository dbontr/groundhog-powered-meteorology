import { parseCSV } from "./lib/stats.js";
import { indexOutcomes, indexPredictions, predictionToOutcome } from "./lib/backtest.js";
import {
  buildDynamicSuperModel,
  computeDynamicSuperNowcast,
  computeHistoricalClimatologyBacktest
} from "./lib/fusion.js";

const $ = (id) => document.getElementById(id);

const TARGET_BASE = "US_CONUS_FEBMAR_MEAN_ANOM";
const TARGET_MARCH = "US_CONUS_MAR_ANOM";
const MIN_OBS = 20;
const MIN_BACKTEST_GH = 20;
const CLIMATOLOGY_WINDOW_YEARS = 15;
const LEADERBOARD_DEFAULT_MIN_OBS = MIN_OBS;

async function loadJson(url) {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} (${url})`);
  return await res.json();
}

async function loadText(url) {
  const res = await fetch(url, { cache: "no-cache" });
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

function computeLeaderboard(predByYear, outcomes, groundhogDir, baselineByYear, minObs = MIN_OBS) {
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
    const baselinePred = baselineByYear.get(year)?.pred ?? "";
    for (const p of preds) {
      const slug = p.groundhogSlug;
      if (!slug) continue;
      if (!stats.has(slug)) stats.set(slug, { n: 0, k: 0, baselineK: 0 });
      const s = stats.get(slug);
      s.n += 1;
      const predOut = predictionToOutcome(!!p.shadow);
      if (predOut === actual) s.k += 1;
      if (baselinePred === actual) s.baselineK += 1;
    }
  }

  const rows = Array.from(stats.entries()).map(([slug, s]) => {
    const accuracy = s.n ? s.k / s.n : Number.NaN;
    const adjustedAccuracy = (s.k + 2) / (s.n + 4);
    const baselineAccuracy = (s.baselineK + 2) / (s.n + 4);
    return {
      slug,
      name: nameBySlug.get(slug) ?? slug,
      n: s.n,
      k: s.k,
      accuracy,
      adjustedAccuracy,
      baselineAccuracy,
      skill: adjustedAccuracy - baselineAccuracy
    };
  }).filter(r => r.n >= minObs);

  rows.sort((a, b) => {
    if (b.skill !== a.skill) return b.skill - a.skill;
    if (b.adjustedAccuracy !== a.adjustedAccuracy) return b.adjustedAccuracy - a.adjustedAccuracy;
    if (b.n !== a.n) return b.n - a.n;
    return a.name.localeCompare(b.name);
  });

  return rows;
}

function countTotalGroundhogs(groundhogDir, predByYear) {
  const list = groundhogDir?.groundhogs;
  if (Array.isArray(list) && list.length) return list.length;
  const seen = new Set();
  for (const preds of predByYear.values()) {
    for (const p of preds) {
      if (p?.groundhogSlug) seen.add(p.groundhogSlug);
    }
  }
  return seen.size;
}

function renderLeaderboard(rows) {
  const table = $("leaderboard");
  if (!table) return;

  const compact = window.matchMedia("(max-width: 520px)").matches;
  const labels = {
    rank: compact ? "#" : "Rank",
    groundhog: "Groundhog",
    accuracy: compact ? "Acc" : "Accuracy",
    skill: compact ? "Skill" : "Climate Skill",
    obs: compact ? "Obs" : "Observations"
  };

  if (!rows.length) {
    table.innerHTML = `
      <thead>
        <tr>
          <th class="rank">${labels.rank}</th>
          <th>${labels.groundhog}</th>
          <th>${labels.accuracy}</th>
          <th>${labels.skill}</th>
          <th>${labels.obs}</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td colspan="5">No scored groundhog predictions yet.</td>
        </tr>
      </tbody>
    `;
    return;
  }

  const body = rows.map((g, idx) => {
    const accuracy = Number.isFinite(g.accuracy) ? g.accuracy * 100 : Number.NaN;
    const skill = Number.isFinite(g.skill) ? g.skill * 100 : Number.NaN;
    const skillLabel = Number.isFinite(skill) && skill > 0 ? `+${fmtPctValue(skill, 1)}` : fmtPctValue(skill, 1);
    return `
      <tr>
        <td class="rank">${String(idx + 1).padStart(2, "0")}</td>
        <td>${g.name}</td>
        <td class="accuracy">${fmtPctValue(accuracy, 1)}</td>
        <td>${skillLabel}</td>
        <td>${g.n}</td>
      </tr>
    `;
  }).join("");

  table.innerHTML = `
    <thead>
      <tr>
        <th class="rank">${labels.rank}</th>
        <th>${labels.groundhog}</th>
        <th>${labels.accuracy}</th>
        <th>${labels.skill}</th>
        <th>${labels.obs}</th>
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

    const leaderboardYears = Array.from(predByYear.keys())
      .filter((y) => outcomes.has(`${TARGET_BASE}:${y}`));
    const scoredYears = leaderboardYears
      .filter((y) => hasMinGroundhogs(predByYear, y));
    const minYear = leaderboardYears.length ? Math.min(...leaderboardYears) : null;
    const maxYear = leaderboardYears.length ? Math.max(...leaderboardYears) : null;
    const baseline = computeHistoricalClimatologyBacktest(
      outcomes,
      TARGET_BASE,
      scoredYears,
      { windowYears: CLIMATOLOGY_WINDOW_YEARS }
    );
    const leaderboardBaseline = computeHistoricalClimatologyBacktest(
      outcomes,
      TARGET_BASE,
      leaderboardYears,
      { windowYears: CLIMATOLOGY_WINDOW_YEARS }
    );
    const baselineByYear = new Map(
      leaderboardBaseline.rows.map((row) => [row.year, row])
    );
    const leaderboardButton = $("toggleNewbies");
    let allowNewbies = false;
    let latestPredCount = null;
    const updateVoterDetail = () => {
      const detail = $("voterDetail");
      if (!detail) return;
      const parts = [];
      if (Number.isFinite(latestPredCount)) parts.push(`Latest year predictions: ${latestPredCount}.`);
      detail.textContent = parts.join(" ");
    };
    const buildLeaderboardMeta = (minObs) => {
      const compact = window.matchMedia("(max-width: 520px)").matches;
      const obsText = minObs <= 1
        ? "Min observations: none (newbies included)."
        : `Min observations: ${minObs}.`;
      const yearText = scoredYears.length ? ` Scored years: ${minYear}–${maxYear}.` : "";
      const keyText = compact ? " Key: #=Rank, Acc=Accuracy, Skill=adjusted accuracy minus climatology, Obs=Observations." : "";
      return `Skill compares each animal with the ${CLIMATOLOGY_WINDOW_YEARS}-year climatology during the same active years. ${obsText}${yearText} The model backtest requires at least ${MIN_BACKTEST_GH} animals per year.${keyText}`;
    };
    const updateLeaderboard = () => {
      const minObs = allowNewbies ? 1 : LEADERBOARD_DEFAULT_MIN_OBS;
      const rows = computeLeaderboard(
        predByYear,
        outcomes,
        groundhogDir,
        baselineByYear,
        minObs
      );
      renderLeaderboard(rows);
      $("leaderboardMeta").textContent = buildLeaderboardMeta(minObs);
      updateVoterDetail();
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

    const compactQuery = window.matchMedia("(max-width: 520px)");
    const refreshLeaderboard = () => updateLeaderboard();
    if (compactQuery.addEventListener) {
      compactQuery.addEventListener("change", refreshLeaderboard);
    } else {
      window.addEventListener("resize", refreshLeaderboard);
    }

    const isSample = String(predObj.updatedAt || "").includes("SAMPLE");
    if (isSample) {
      setStatus("Sample data loaded — run npm run update:predictions for the full leaderboard.");
    }

    const model = buildDynamicSuperModel(predByYear, outcomes, TARGET_BASE);
    if (!model) {
      setStatus("Backtest unavailable — model could not be evaluated.");
      return;
    }

    const baselineAccuracy = $("baselineAccuracy");
    if (baselineAccuracy) baselineAccuracy.textContent = fmtPct(baseline.accuracy);
    const baselineDetail = $("baselineDetail");
    if (baselineDetail) {
      baselineDetail.textContent = `15-year rolling prior; balanced accuracy ${fmtPct(baseline.balancedAccuracy)}; Brier score ${baseline.brierScore.toFixed(3)}.`;
    }
    const balancedAccuracy = $("balancedAccuracy");
    if (balancedAccuracy) balancedAccuracy.textContent = fmtPct(model.backtest.balancedAccuracy);
    const modelDetail = $("modelDetail");
    if (modelDetail) {
      modelDetail.textContent = `Nested walk-forward, ${model.backtest.backtestN} years; Brier score ${model.backtest.brierScore.toFixed(3)}.`;
    }

    const totalGroundhogs = countTotalGroundhogs(groundhogDir, predByYear);
    const voterCount = $("voterCount");
    if (voterCount) voterCount.textContent = `${totalGroundhogs}`;

    const nowcast = computeDynamicSuperNowcast(predByYear, model);
    if (!nowcast || !nowcast.pred) {
      setStatus("No prediction data available.");
      return;
    }

    const indicator = $("indicator");
    indicator.textContent = outcomeLabel(nowcast.pred);
    document.body.dataset.outcome = nowcast.pred;

    const forecastProbability = nowcast.pred === "EARLY_SPRING"
      ? nowcast.probability
      : 1 - nowcast.probability;
    const forecastProbabilityEl = $("forecastProbability");
    if (forecastProbabilityEl) forecastProbabilityEl.textContent = fmtPct(forecastProbability, 1);
    const algoAccuracy = $("algoAccuracy");
    if (algoAccuracy) algoAccuracy.textContent = fmtPct(model.backtest.accuracy);
    $("callYear").textContent = `Forecast for ${nowcast.latestYear}`;
    $("predictionYear").textContent = `${nowcast.latestYear}`;
    const predCount = nowcast.totalPreds || nowcast.used;
    if (Number.isFinite(predCount)) {
      latestPredCount = predCount;
      updateVoterDetail();
    }

    const contribution = fmtPct(nowcast.groundhogWeight ?? 0);
    $("meta").textContent = `Groundhog contribution: ${contribution}. The climate guard increases animal weight only after prior-year Brier-score improvement. Active profile: ${nowcast.profileId}.`;

    if (!isSample) setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(String(err));
  }
}

run();
