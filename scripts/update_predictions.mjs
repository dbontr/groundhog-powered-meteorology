import { promises as fs } from "node:fs";
import path from "node:path";

const API = "https://groundhog-day.com/api/v1";
const OUT_DIR = path.join(process.cwd(), "docs", "data");

const START_YEAR = 1887;
const END_YEAR = new Date().getFullYear();

function sleep(ms){ return new Promise(r=>setTimeout(r, ms)); }

async function fetchJson(url, tries = 3) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, { headers: { "accept": "application/json" } });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return await res.json();
    } catch (e) {
      lastErr = e;
      await sleep(250 * (i + 1));
    }
  }
  throw lastErr;
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });

  console.log(`Fetching groundhog list from ${API}/groundhogs ...`);
  const gh = await fetchJson(`${API}/groundhogs`);
  const groundhogs = Array.isArray(gh.groundhogs) ? gh.groundhogs : (Array.isArray(gh) ? gh : []);
  if (!groundhogs.length) {
    throw new Error("Unexpected /groundhogs response shape; got no groundhogs.");
  }

  // Save a trimmed groundhog directory (enough for UI + matching).
  const groundhogDir = groundhogs.map(g => ({
    id: g.id ?? null,
    slug: g.slug ?? null,
    name: g.name ?? null,
    shortName: g.shortName ?? null,
    region: g.region ?? null,
    country: g.country ?? null,
    state: g.state ?? null,
    city: g.city ?? null,
    latitude: g.latitude ?? null,
    longitude: g.longitude ?? null,
    source: g.source ?? null,
    predictionsCount: g.predictionsCount ?? null
  })).filter(g => g.slug);

  await fs.writeFile(path.join(OUT_DIR, "groundhogs.json"), JSON.stringify({ updatedAt: new Date().toISOString(), groundhogs: groundhogDir }, null, 2));

  // If the /groundhogs endpoint already includes predictions, use them.
  const hasPredictionsInline = groundhogs.some(g => Array.isArray(g.predictions) && g.predictions.length);
  let predictions = [];

  if (hasPredictionsInline) {
    console.log("Found predictions embedded in /groundhogs response. Extracting...");
    for (const g of groundhogs) {
      if (!Array.isArray(g.predictions)) continue;
      for (const p of g.predictions) {
        predictions.push({
          year: p.year,
          shadow: !!p.shadow,
          groundhogSlug: g.slug,
          groundhogName: g.name,
          details: p.details ?? null,
          source: p.source ?? g.source ?? null
        });
      }
    }
  } else {
    console.log("No embedded predictions found. Pulling per-year predictions...");
    const years = [];
    for (let y = START_YEAR; y <= END_YEAR; y++) years.push(y);

    const concurrency = 6;
    let idx = 0;

    async function worker() {
      while (idx < years.length) {
        const y = years[idx++];
        const url = `${API}/predictions?year=${y}`;
        try {
          const data = await fetchJson(url);
          const rows = Array.isArray(data.predictions) ? data.predictions : [];
          for (const r of rows) {
            const g = r.groundhog || {};
            predictions.push({
              year: r.year ?? y,
              shadow: !!r.shadow,
              groundhogSlug: g.slug ?? null,
              groundhogName: g.name ?? null,
              details: r.details ?? null,
              source: r.source ?? g.source ?? null
            });
          }
          process.stdout.write(".");
        } catch (e) {
          console.warn(`\n⚠️  ${y}: ${e}`);
        }
      }
    }

    await Promise.all(Array.from({ length: concurrency }, worker));
    process.stdout.write("\n");
  }

  // Keep only the fields we need.
  predictions = predictions
    .filter(p => Number.isFinite(+p.year) && p.groundhogSlug)
    .sort((a, b) => (a.year - b.year) || a.groundhogSlug.localeCompare(b.groundhogSlug));

  console.log(`Writing ${predictions.length.toLocaleString()} predictions...`);
  await fs.writeFile(path.join(OUT_DIR, "predictions.json"),
    JSON.stringify({ updatedAt: new Date().toISOString(), predictions }, null, 2)
  );

  console.log("✓ done");
}

main().catch((e) => {
  console.error("ERROR:", e);
  process.exit(1);
});
