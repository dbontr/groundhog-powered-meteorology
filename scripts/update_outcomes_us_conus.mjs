import { promises as fs } from "node:fs";
import path from "node:path";

const OUT_DIR = path.join(process.cwd(), "docs", "data");
const END_YEAR = new Date().getFullYear();
const START_YEAR = 1887;

// --- Classification knobs ---
// Groundhogs predict ONLY 2 outcomes, so our *scoring truth* is binary:
//   mean_anom > 0  => EARLY_SPRING
//   mean_anom <= 0 => LONG_WINTER
//
// We *also* compute a winter-intensity tag INSIDE the LONG_WINTER bucket:
//   winter_bucket = EARLY_WINTER if mean_anom <= -WINTER_THRESHOLD_F
//                 = NORMAL_WINTER otherwise
// (Blank for EARLY_SPRING years.)
//
// You can override via env var in GitHub Actions.
const WINTER_THRESHOLD_F = Number.parseFloat(process.env.WINTER_THRESHOLD_F ?? "1.0");

// NOAA Climate-at-a-Glance “CSV-style” URLs are widely used in coursework/tutorials.
// If NOAA changes the URL format, adjust `urlFor(month)` below.
function urlFor(month) {
  // 110 = contiguous U.S. (CONUS) in many CAG endpoints.
  // parameter = tavg (average temperature), timeScale = 1 (monthly), month = {2,3}.
  // base_prd=true with 1901–2000 baseline yields anomalies relative to that base period.
  return `https://www.ncdc.noaa.gov/cag/national/time-series/110-tavg-1-${month}-1895-${END_YEAR}.csv?base_prd=true&begbaseyear=1901&endbaseyear=2000`;
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.text();
}

function parseCagCsv(text) {
  // Robust-ish: skip non-data lines; accept rows like "1895, -0.12"
  const rows = new Map();
  for (const line of text.split(/\r?\n/)) {
    const m = line.match(/^\s*(\d{4})\s*,\s*(-?\d+(\.\d+)?)/);
    if (!m) continue;
    rows.set(+m[1], +m[2]);
  }
  return rows;
}

function outcomeFromMean(meanAnom) {
  if (!Number.isFinite(meanAnom)) return "";
  return meanAnom > 0 ? "EARLY_SPRING" : "LONG_WINTER";
}

function winterBucketFromMean(meanAnom) {
  if (!Number.isFinite(meanAnom)) return "";
  // Only meaningful when the binary outcome is LONG_WINTER.
  if (meanAnom > 0) return "";
  const thr = Math.abs(WINTER_THRESHOLD_F);
  return (meanAnom <= -thr) ? "EARLY_WINTER" : "NORMAL_WINTER";
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });

  console.log("Fetching NOAA CAG Feb anomalies…");
  const febText = await fetchText(urlFor(2));
  console.log("Fetching NOAA CAG Mar anomalies…");
  const marText = await fetchText(urlFor(3));

  const feb = parseCagCsv(febText);
  const mar = parseCagCsv(marText);

  const lines = [];
  lines.push("# Outcomes definition: CONUS mean(Feb_anomaly, Mar_anomaly) relative to 1901–2000 baseline.");
  lines.push("# outcome  (binary)  = EARLY_SPRING if mean_anom > 0 else LONG_WINTER");
  lines.push(`# winter_bucket (only when outcome=LONG_WINTER) = EARLY_WINTER if mean_anom <= -${WINTER_THRESHOLD_F}F else NORMAL_WINTER`);
  lines.push("year,target,feb_anom,mar_anom,mean_anom,outcome,winter_bucket");

  for (let y = START_YEAR; y <= END_YEAR; y++) {
    const fa = feb.get(y);
    const ma = mar.get(y);
    const mean = (Number.isFinite(fa) && Number.isFinite(ma)) ? (fa + ma) / 2 : NaN;
    const out = outcomeFromMean(mean);
    const wb = winterBucketFromMean(mean);
    lines.push([
      y,
      "US_CONUS_FEBMAR_MEAN_ANOM",
      Number.isFinite(fa) ? fa.toFixed(3) : "",
      Number.isFinite(ma) ? ma.toFixed(3) : "",
      Number.isFinite(mean) ? mean.toFixed(3) : "",
      out,
      wb
    ].join(","));
  }

  const outPath = path.join(OUT_DIR, "outcomes.csv");
  await fs.writeFile(outPath, lines.join("\n") + "\n", "utf8");
  console.log(`✓ wrote ${path.relative(process.cwd(), outPath)}`);
}

main().catch((e) => {
  console.error("ERROR:", e);
  console.error("\nIf this fails, NOAA may have changed the CAG CSV endpoint format. Update urlFor(month).");
  process.exit(1);
});
