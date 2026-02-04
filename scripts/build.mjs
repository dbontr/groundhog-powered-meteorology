import { promises as fs } from "node:fs";
import path from "node:path";

const root = process.cwd();
const dataDir = path.join(root, "docs", "data");

async function ensureCopy(src, dst) {
  try {
    await fs.access(dst);
    console.log(`✓ ${path.relative(root, dst)} exists`);
  } catch {
    await fs.copyFile(src, dst);
    console.log(`→ seeded ${path.relative(root, dst)} from ${path.relative(root, src)}`);
  }
}

await fs.mkdir(dataDir, { recursive: true });

await ensureCopy(path.join(dataDir, "predictions.sample.json"), path.join(dataDir, "predictions.json"));
await ensureCopy(path.join(dataDir, "groundhogs.sample.json"), path.join(dataDir, "groundhogs.json"));
await ensureCopy(path.join(dataDir, "outcomes.sample.csv"), path.join(dataDir, "outcomes.csv"));
