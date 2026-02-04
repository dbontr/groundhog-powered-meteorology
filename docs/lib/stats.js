export function wilsonCI(k, n, z = 1.96) {
  if (n <= 0) return { lo: 0, hi: 1, center: 0.5, half: 0.5 };
  const p = k / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const half = (z / denom) * Math.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n)));
  return { lo: Math.max(0, center - half), hi: Math.min(1, center + half), center, half };
}

export function clamp(x, a, b){ return Math.max(a, Math.min(b, x)); }

export function parseCSV(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim().length && !l.startsWith("#"));
  if (!lines.length) return [];
  const headers = lines[0].split(",").map(s => s.trim());
  const rows = [];
  for (const line of lines.slice(1)) {
    const parts = line.split(","); // simple CSV; good enough for our controlled files
    const obj = {};
    for (let i = 0; i < headers.length; i++) obj[headers[i]] = (parts[i] ?? "").trim();
    rows.push(obj);
  }
  return rows;
}
