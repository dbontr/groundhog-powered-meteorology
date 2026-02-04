import os
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "data"
END_YEAR = datetime.utcnow().year
START_YEAR = 1887

WINTER_THRESHOLD_F = float(os.environ.get("WINTER_THRESHOLD_F", "1.0"))


def url_for(month, end_year):
    return (
        "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
        f"national/time-series/110/tavg/1/{month}/1895-{end_year}/data.csv"
        "?base_prd=true&begbaseyear=1901&endbaseyear=2000"
    )


def fetch_text(url):
    with urlopen(url) as resp:
        return resp.read().decode("utf-8")


def parse_cag_csv(text):
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts or parts[0].lower() == "date":
            continue
        date_str = parts[0]
        if len(date_str) < 4 or not date_str[:4].isdigit():
            continue
        year = int(date_str[:4])
        val_str = parts[2] if len(parts) >= 3 and parts[2] else (parts[1] if len(parts) >= 2 else "")
        if not val_str:
            continue
        try:
            rows[year] = float(val_str)
        except ValueError:
            continue
    return rows


def outcome_from_mean(mean_anom):
    if mean_anom is None:
        return ""
    return "EARLY_SPRING" if mean_anom > 0 else "LONG_WINTER"


def winter_bucket_from_mean(mean_anom):
    if mean_anom is None:
        return ""
    if mean_anom > 0:
        return ""
    thr = abs(WINTER_THRESHOLD_F)
    return "EARLY_WINTER" if mean_anom <= -thr else "NORMAL_WINTER"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    end_year = END_YEAR
    feb_text = None
    mar_text = None
    while end_year >= 1895:
        try:
            print(f"Fetching NOAA CAG Feb anomalies (end year {end_year})…")
            feb_text = fetch_text(url_for(2, end_year))
            print(f"Fetching NOAA CAG Mar anomalies (end year {end_year})…")
            mar_text = fetch_text(url_for(3, end_year))
            break
        except HTTPError as exc:
            if exc.code != 404:
                raise
            end_year -= 1
    if feb_text is None or mar_text is None:
        raise RuntimeError("Could not resolve a valid NOAA CAG endpoint.")

    feb = parse_cag_csv(feb_text)
    mar = parse_cag_csv(mar_text)

    lines = []
    lines.append("# Outcomes definition: CONUS mean(Feb_anomaly, Mar_anomaly) relative to 1901–2000 baseline.")
    lines.append("# outcome  (binary)  = EARLY_SPRING if mean_anom > 0 else LONG_WINTER")
    lines.append(
        f"# winter_bucket (only when outcome=LONG_WINTER) = EARLY_WINTER if mean_anom <= -{WINTER_THRESHOLD_F}F else NORMAL_WINTER"
    )
    lines.append("year,target,feb_anom,mar_anom,mean_anom,outcome,winter_bucket")

    for y in range(START_YEAR, end_year + 1):
        fa = feb.get(y)
        ma = mar.get(y)
        if fa is None or ma is None:
            mean = None
        else:
            mean = (fa + ma) / 2
        out = outcome_from_mean(mean)
        wb = winter_bucket_from_mean(mean)
        lines.append(",".join([
            str(y),
            "US_CONUS_FEBMAR_MEAN_ANOM",
            f"{fa:.3f}" if fa is not None else "",
            f"{ma:.3f}" if ma is not None else "",
            f"{mean:.3f}" if mean is not None else "",
            out,
            wb
        ]))

    out_path = OUT_DIR / "outcomes.csv"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ wrote {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
