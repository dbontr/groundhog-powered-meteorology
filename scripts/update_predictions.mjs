import { promises as fs } from "node:fs";
import path from "node:path";

const API = "https://groundhog-day.com/api/v1";
const OUT_DIR = path.join(process.cwd(), "docs", "data");

const START_YEAR = 1887;
const END_YEAR = new Date().getFullYear();

function sleep(ms){ return new Promise(r=>setTimeout(r, ms)); }

function isExplicitShadow(value) {
  return value === 0 || value === 1 || typeof value === "boolean";
}

function parseCoordinates(value) {
  if (typeof value !== "string") return { latitude: null, longitude: null };
  const [latitude, longitude] = value.split(",").map(Number);
  return {
    latitude: Number.isFinite(latitude) ? latitude : null,
    longitude: Number.isFinite(longitude) ? longitude : null
  };
}

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
  const allForecasters = Array.isArray(gh.groundhogs)
    ? gh.groundhogs
    : (Array.isArray(gh) ? gh : []);
  if (!allForecasters.length) {
    throw new Error("Unexpected /groundhogs response shape; got no forecasters.");
  }

  // This project is intentionally powered by literal groundhogs only.
  const groundhogs = allForecasters.filter(g => Number(g.isGroundhog) === 1);
  if (!groundhogs.length) {
    throw new Error("The API returned no entries marked isGroundhog=1.");
  }
  const groundhogSlugs = new Set(groundhogs.map(g => g.slug).filter(Boolean));

  // Save a compact groundhog-only directory for the site.
  const groundhogDir = groundhogs.map(g => {
    const coordinates = parseCoordinates(g.coordinates);
    return {
      id: g.id ?? null,
      slug: g.slug ?? null,
      name: g.name ?? null,
      shortName: g.shortname ?? g.shortName ?? null,
      region: g.region ?? null,
      country: g.country ?? null,
      state: g.state ?? null,
      city: g.city ?? null,
      latitude: g.latitude ?? coordinates.latitude,
      longitude: g.longitude ?? coordinates.longitude,
      source: g.source ?? null,
      isGroundhog: true,
      type: g.type ?? "Groundhog",
      active: Boolean(g.active),
      predictionsCount: g.predictionsCount ?? null
    };
  }).filter(g => g.slug);

  await fs.writeFile(
    path.join(OUT_DIR, "groundhogs.json"),
    JSON.stringify({
      updatedAt: new Date().toISOString(),
      groundhogsOnly: true,
      groundhogs: groundhogDir
    }, null, 2)
  );

  // If the /groundhogs endpoint already includes predictions, use them.
  const hasPredictionsInline = groundhogs.some(g => Array.isArray(g.predictions) && g.predictions.length);
  let predictions = [];

  if (hasPredictionsInline) {
    console.log("Found predictions embedded in /groundhogs response. Extracting...");
    for (const g of groundhogs) {
      if (!Array.isArray(g.predictions)) continue;
      for (const p of g.predictions) {
        if (!isExplicitShadow(p.shadow)) continue;
        predictions.push({
          year: p.year,
          shadow: p.shadow === 1 || p.shadow === true,
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
            if (!groundhogSlugs.has(g.slug) || !isExplicitShadow(r.shadow)) continue;
            predictions.push({
              year: r.year ?? y,
              shadow: r.shadow === 1 || r.shadow === true,
              groundhogSlug: g.slug,
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
    .filter(p => Number.isFinite(+p.year)
      && groundhogSlugs.has(p.groundhogSlug)
      && typeof p.shadow === "boolean")
    .sort((a, b) => (a.year - b.year) || a.groundhogSlug.localeCompare(b.groundhogSlug));

  console.log(`Writing ${predictions.length.toLocaleString()} verified groundhog predictions...`);
  await fs.writeFile(
    path.join(OUT_DIR, "predictions.json"),
    JSON.stringify({
      updatedAt: new Date().toISOString(),
      groundhogsOnly: true,
      excludesMissingPredictions: true,
      predictions
    }, null, 2)
  );

  console.log("✓ done");
}

main().catch((e) => {
  console.error("ERROR:", e);
  process.exit(1);
});
