import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

test("browser code only references existing element ids", async () => {
  const [app, html] = await Promise.all([
    readFile("docs/app.js", "utf8"),
    readFile("docs/index.html", "utf8")
  ]);
  const references = [...app.matchAll(/\$\("([^"]+)"\)/g)].map((match) => match[1]);
  const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map((match) => match[1]));
  const missing = [...new Set(references.filter((id) => !ids.has(id)))];
  assert.deepEqual(missing, []);
});
