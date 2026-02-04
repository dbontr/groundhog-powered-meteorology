import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://groundhog-day.com/api/v1"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "data"

START_YEAR = 1887
END_YEAR = datetime.utcnow().year


def sleep(ms):
    time.sleep(ms / 1000)


def fetch_json(url, tries=3, timeout=30):
    last_err = None
    for i in range(tries):
        try:
            req = Request(url, headers={"accept": "application/json"})
            with urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:
            last_err = exc
            sleep(250 * (i + 1))
    raise last_err


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching groundhog list from {API}/groundhogs ...")
    gh = fetch_json(f"{API}/groundhogs")
    if isinstance(gh, dict):
        groundhogs = gh.get("groundhogs") or []
    elif isinstance(gh, list):
        groundhogs = gh
    else:
        groundhogs = []

    if not groundhogs:
        raise RuntimeError("Unexpected /groundhogs response shape; got no groundhogs.")

    groundhog_dir = []
    for g in groundhogs:
        slug = g.get("slug")
        if not slug:
            continue
        groundhog_dir.append({
            "id": g.get("id"),
            "slug": slug,
            "name": g.get("name"),
            "shortName": g.get("shortName"),
            "region": g.get("region"),
            "country": g.get("country"),
            "state": g.get("state"),
            "city": g.get("city"),
            "latitude": g.get("latitude"),
            "longitude": g.get("longitude"),
            "source": g.get("source"),
            "predictionsCount": g.get("predictionsCount")
        })

    groundhog_path = OUT_DIR / "groundhogs.json"
    with groundhog_path.open("w", encoding="utf-8") as f:
        json.dump({
            "updatedAt": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "groundhogs": groundhog_dir
        }, f, indent=2)

    has_inline = any(isinstance(g.get("predictions"), list) and g.get("predictions") for g in groundhogs)
    predictions = []

    if has_inline:
        print("Found predictions embedded in /groundhogs response. Extracting...")
        for g in groundhogs:
            preds = g.get("predictions")
            if not isinstance(preds, list):
                continue
            for p in preds:
                predictions.append({
                    "year": p.get("year"),
                    "shadow": bool(p.get("shadow")),
                    "groundhogSlug": g.get("slug"),
                    "groundhogName": g.get("name"),
                    "details": p.get("details"),
                    "source": p.get("source") or g.get("source")
                })
    else:
        print("No embedded predictions found. Pulling per-year predictions...")
        years = list(range(START_YEAR, END_YEAR + 1))

        def fetch_year(y):
            url = f"{API}/predictions?year={y}"
            data = fetch_json(url)
            rows = data.get("predictions") if isinstance(data, dict) else []
            preds = []
            for r in rows or []:
                g = r.get("groundhog") or {}
                preds.append({
                    "year": r.get("year", y),
                    "shadow": bool(r.get("shadow")),
                    "groundhogSlug": g.get("slug"),
                    "groundhogName": g.get("name"),
                    "details": r.get("details"),
                    "source": r.get("source") or g.get("source")
                })
            return preds

        errors = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            future_map = {pool.submit(fetch_year, y): y for y in years}
            for fut in as_completed(future_map):
                y = future_map[fut]
                try:
                    predictions.extend(fut.result())
                    sys.stdout.write(".")
                    sys.stdout.flush()
                except Exception as exc:
                    errors += 1
                    sys.stderr.write(f"\n⚠️  {y}: {exc}\n")
            sys.stdout.write("\n")

        if errors:
            print(f"Completed with {errors} failed year(s).", file=sys.stderr)

    predictions = [
        p for p in predictions
        if isinstance(p.get("year"), int) and p.get("groundhogSlug")
    ]
    predictions.sort(key=lambda p: (p["year"], p["groundhogSlug"]))

    print(f"Writing {len(predictions):,} predictions...")
    pred_path = OUT_DIR / "predictions.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump({
            "updatedAt": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "predictions": predictions
        }, f, indent=2)

    print("✓ done")


if __name__ == "__main__":
    main()
