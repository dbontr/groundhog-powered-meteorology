#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

TARGET_BASE = "US_CONUS_FEBMAR_MEAN_ANOM"
TARGET_MARCH = "US_CONUS_MAR_ANOM"
MIN_BACKTEST_GH = 20

GOAL_ACCURACY = 0.70

FUSION_CONFIGS = [
    {
        "id": "spark",
        "halfLifeYears": 6,
        "windowYears": 12,
        "minObs": 6,
        "nBoost": 0.45,
        "priorA": 2,
        "priorB": 2,
        "wBayes": 1.2,
        "wDecay": 1.6,
        "wWindow": 0.9,
        "wStability": 0.7,
        "wEvidence": 0.5,
        "wTrend": 0.25,
        "contrarian": True,
    },
    {
        "id": "balanced",
        "halfLifeYears": 10,
        "windowYears": 18,
        "minObs": 8,
        "nBoost": 0.55,
        "priorA": 3,
        "priorB": 3,
        "wBayes": 1.4,
        "wDecay": 1.2,
        "wWindow": 1.0,
        "wStability": 0.8,
        "wEvidence": 0.6,
        "wTrend": 0.2,
        "contrarian": True,
    },
    {
        "id": "stable",
        "halfLifeYears": 16,
        "windowYears": 26,
        "minObs": 10,
        "nBoost": 0.6,
        "priorA": 4,
        "priorB": 4,
        "wBayes": 1.6,
        "wDecay": 0.9,
        "wWindow": 1.2,
        "wStability": 1.2,
        "wEvidence": 0.7,
        "wTrend": 0.1,
        "contrarian": False,
    },
    {
        "id": "recency",
        "halfLifeYears": 4,
        "windowYears": 9,
        "minObs": 5,
        "nBoost": 0.4,
        "priorA": 2,
        "priorB": 2,
        "wBayes": 0.9,
        "wDecay": 1.8,
        "wWindow": 0.8,
        "wStability": 0.5,
        "wEvidence": 0.4,
        "wTrend": 0.35,
        "contrarian": True,
    },
    {
        "id": "legacy",
        "halfLifeYears": 22,
        "windowYears": 32,
        "minObs": 12,
        "nBoost": 0.7,
        "priorA": 5,
        "priorB": 5,
        "wBayes": 1.7,
        "wDecay": 0.6,
        "wWindow": 1.3,
        "wStability": 1.0,
        "wEvidence": 0.9,
        "wTrend": 0.05,
        "contrarian": False,
    },
    {
        "id": "stability",
        "halfLifeYears": 12,
        "windowYears": 20,
        "minObs": 8,
        "nBoost": 0.5,
        "priorA": 3,
        "priorB": 3,
        "wBayes": 1.3,
        "wDecay": 1.0,
        "wWindow": 0.9,
        "wStability": 1.4,
        "wEvidence": 0.5,
        "wTrend": 0.15,
        "contrarian": False,
    },
    {
        "id": "window",
        "halfLifeYears": 9,
        "windowYears": 10,
        "minObs": 5,
        "nBoost": 0.45,
        "priorA": 2,
        "priorB": 2,
        "wBayes": 1.0,
        "wDecay": 1.0,
        "wWindow": 1.6,
        "wStability": 0.6,
        "wEvidence": 0.4,
        "wTrend": 0.2,
        "contrarian": True,
    },
    {
        "id": "momentum",
        "halfLifeYears": 5,
        "windowYears": 8,
        "minObs": 4,
        "nBoost": 0.35,
        "priorA": 2,
        "priorB": 2,
        "wBayes": 0.8,
        "wDecay": 1.7,
        "wWindow": 0.7,
        "wStability": 0.5,
        "wEvidence": 0.3,
        "wTrend": 0.6,
        "contrarian": True,
    },
    {
        "id": "evidence",
        "halfLifeYears": 12,
        "windowYears": 24,
        "minObs": 12,
        "nBoost": 0.8,
        "priorA": 4,
        "priorB": 4,
        "wBayes": 1.2,
        "wDecay": 0.8,
        "wWindow": 0.7,
        "wStability": 1.0,
        "wEvidence": 1.1,
        "wTrend": 0.05,
        "contrarian": False,
    },
]

TUNING_SETS = [
    {
        "id": "balanced",
        "gate": 0.56,
        "topK": 6,
        "minModels": 3,
        "minUsedRatio": 0.45,
        "minConfigYears": 8,
        "rankWindowYears": 24,
        "rankDecayHalfLife": 12,
        "stack": {"steps": 260, "lr": 0.22, "l2": 0.06, "minTrain": 12, "decayHalfLife": 12, "windowYears": 30},
        "blend": {"minTrain": 8, "windowYears": 26, "decayHalfLife": 10, "stabilityBoost": 0.8},
    },
    {
        "id": "aggressive",
        "gate": 0.60,
        "topK": 5,
        "minModels": 2,
        "minUsedRatio": 0.35,
        "minConfigYears": 6,
        "rankWindowYears": 18,
        "rankDecayHalfLife": 8,
        "stack": {"steps": 300, "lr": 0.26, "l2": 0.05, "minTrain": 10, "decayHalfLife": 8, "windowYears": 20},
        "blend": {"minTrain": 7, "windowYears": 16, "decayHalfLife": 6, "stabilityBoost": 0.5},
    },
    {
        "id": "steady",
        "gate": 0.54,
        "topK": 7,
        "minModels": 4,
        "minUsedRatio": 0.55,
        "minConfigYears": 10,
        "rankWindowYears": 32,
        "rankDecayHalfLife": 16,
        "stack": {"steps": 240, "lr": 0.18, "l2": 0.08, "minTrain": 14, "decayHalfLife": 16, "windowYears": 35},
        "blend": {"minTrain": 9, "windowYears": 30, "decayHalfLife": 14, "stabilityBoost": 1.1},
    },
    {
        "id": "recency",
        "gate": 0.58,
        "topK": 4,
        "minModels": 2,
        "minUsedRatio": 0.35,
        "minConfigYears": 6,
        "rankWindowYears": 12,
        "rankDecayHalfLife": 4,
        "stack": {"steps": 280, "lr": 0.24, "l2": 0.05, "minTrain": 9, "decayHalfLife": 6, "windowYears": 12},
        "blend": {"minTrain": 7, "windowYears": 10, "decayHalfLife": 4, "stabilityBoost": 0.3},
    },
    {
        "id": "conservative",
        "gate": 0.52,
        "topK": 8,
        "minModels": 5,
        "minUsedRatio": 0.6,
        "minConfigYears": 12,
        "rankWindowYears": 40,
        "rankDecayHalfLife": 20,
        "stack": {"steps": 220, "lr": 0.16, "l2": 0.1, "minTrain": 16, "decayHalfLife": 20, "windowYears": 40},
        "blend": {"minTrain": 10, "windowYears": 35, "decayHalfLife": 18, "stabilityBoost": 1.2},
    },
]


def parse_csv(text: str) -> list[dict]:
    lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return []
    headers = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        row = {}
        for i, header in enumerate(headers):
            row[header] = (parts[i] if i < len(parts) else "").strip()
        rows.append(row)
    return rows


def build_outcome_rows(outcomes_rows: list[dict]) -> list[dict]:
    rows = [dict(r) for r in outcomes_rows]
    extra = []
    for r in outcomes_rows:
        target = str(r.get("target") or "").strip()
        if target != TARGET_BASE:
            continue
        try:
            mar = float(r.get("mar_anom", "nan"))
        except ValueError:
            continue
        if math.isfinite(mar):
            extra.append({
                "year": r.get("year"),
                "target": TARGET_MARCH,
                "outcome": "EARLY_SPRING" if mar > 0 else "LONG_WINTER",
            })
    rows.extend(extra)
    return rows


def normalize_outcome(x: str) -> str:
    if not x:
        return ""
    s = str(x).strip().upper()
    if s in ("EARLY_SPRING", "ES"):
        return "EARLY_SPRING"
    if s in ("LONG_WINTER", "LW", "MORE_WINTER"):
        return "LONG_WINTER"
    return ""


def index_outcomes(outcome_rows: list[dict]) -> dict[str, str]:
    out = {}
    for r in outcome_rows:
        try:
            year = int(float(r.get("year", "")))
        except ValueError:
            continue
        target = (r.get("target") or "").strip()
        outcome = normalize_outcome(r.get("outcome", ""))
        if not target or not outcome:
            continue
        out[f"{target}:{year}"] = outcome
    return out


def index_predictions(pred_obj: dict) -> dict[int, list[dict]]:
    by_year: dict[int, list[dict]] = {}
    for p in pred_obj.get("predictions", []):
        try:
            year = int(float(p.get("year", "")))
        except ValueError:
            continue
        slug = p.get("groundhogSlug")
        if not slug:
            continue
        by_year.setdefault(year, []).append(p)
    for arr in by_year.values():
        arr.sort(key=lambda p: str(p.get("groundhogSlug")))
    return by_year


def prediction_to_outcome(shadow: bool) -> str:
    return "LONG_WINTER" if shadow else "EARLY_SPRING"


def has_min_groundhogs(pred_by_year: dict[int, list[dict]], year: int, min_count: int = MIN_BACKTEST_GH) -> bool:
    return len(pred_by_year.get(year, [])) >= min_count


def majority_vote(preds: list[dict]) -> dict:
    early = 0
    late = 0
    for p in preds:
        out = prediction_to_outcome(bool(p.get("shadow")))
        if out == "EARLY_SPRING":
            early += 1
        else:
            late += 1
    used = early + late
    if not used:
        return {"pred": "", "certainty": float("nan"), "used": 0}
    pred = "EARLY_SPRING" if early >= late else "LONG_WINTER"
    certainty = max(early, late) / used
    return {"pred": pred, "certainty": certainty, "used": used}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def logit(p: float) -> float:
    q = clamp(p, 1e-6, 1 - 1e-6)
    return math.log(q / (1 - q))


def compute_fusion_stats(pred_by_year: dict[int, list[dict]], outcomes: dict[str, str], target: str, year_exclusive: int, cfg: dict):
    years = sorted([y for y in pred_by_year.keys() if y < year_exclusive and f"{target}:{y}" in outcomes])
    stats: dict[str, dict] = {}
    if not years:
        return {"stats": {}, "maxN": 0}

    split_year = years[len(years) // 2]
    lambda_decay = math.log(2) / max(1e-9, cfg.get("halfLifeYears", 1)) if cfg.get("halfLifeYears") else None
    window_start = year_exclusive - cfg.get("windowYears", 0) if cfg.get("windowYears") else None
    max_n = 0

    for year in years:
        actual = outcomes.get(f"{target}:{year}")
        preds = pred_by_year.get(year, [])
        age = year_exclusive - year
        decay_w = math.exp(-lambda_decay * age) if lambda_decay is not None else 1.0
        in_window = year >= window_start if window_start is not None else True

        for p in preds:
            slug = p.get("groundhogSlug")
            if not slug:
                continue
            s = stats.setdefault(slug, {
                "n": 0,
                "k": 0,
                "nDecay": 0.0,
                "kDecay": 0.0,
                "nWindow": 0,
                "kWindow": 0,
                "nEarly": 0,
                "kEarly": 0,
                "nLate": 0,
                "kLate": 0,
            })
            pred_out = prediction_to_outcome(bool(p.get("shadow")))
            correct = pred_out == actual

            s["n"] += 1
            if correct:
                s["k"] += 1
            s["nDecay"] += decay_w
            if correct:
                s["kDecay"] += decay_w
            if in_window:
                s["nWindow"] += 1
                if correct:
                    s["kWindow"] += 1
            if year < split_year:
                s["nEarly"] += 1
                if correct:
                    s["kEarly"] += 1
            else:
                s["nLate"] += 1
                if correct:
                    s["kLate"] += 1
            if s["n"] > max_n:
                max_n = s["n"]

    derived = {}
    for slug, s in stats.items():
        if not s["n"]:
            continue
        prior_a = cfg.get("priorA", 2)
        prior_b = cfg.get("priorB", 2)
        acc_raw = s["k"] / s["n"]
        acc_bayes = (s["k"] + prior_a) / (s["n"] + prior_a + prior_b)
        acc_decay = (s["kDecay"] + prior_a) / (s["nDecay"] + prior_a + prior_b) if s["nDecay"] else acc_bayes
        acc_window = (s["kWindow"] + prior_a) / (s["nWindow"] + prior_a + prior_b) if s["nWindow"] else acc_bayes
        acc_early = s["kEarly"] / s["nEarly"] if s["nEarly"] else acc_raw
        acc_late = s["kLate"] / s["nLate"] if s["nLate"] else acc_raw
        stability = clamp(1 - abs(acc_early - acc_late), 0, 1)
        trend = acc_late - acc_early
        derived[slug] = {
            "n": s["n"],
            "accRaw": acc_raw,
            "accBayes": acc_bayes,
            "accDecay": acc_decay,
            "accWindow": acc_window,
            "stability": stability,
            "trend": trend,
        }

    return {"stats": derived, "maxN": max_n}


def build_fusion_weights(stats: dict, max_n: int, cfg: dict) -> dict[str, float]:
    weights: dict[str, float] = {}
    denom = math.log1p(max(1, max_n))
    for slug, s in stats.items():
        if s["n"] < cfg.get("minObs", 0):
            continue
        evidence = math.log1p(s["n"]) / denom if denom else 0
        stability = s["stability"] if math.isfinite(s["stability"]) else 0.5
        signal = (
            cfg.get("wBayes", 1) * logit(s["accBayes"])
            + cfg.get("wDecay", 0) * logit(s["accDecay"])
            + cfg.get("wWindow", 0) * logit(s["accWindow"])
            + cfg.get("wStability", 0) * ((stability - 0.5) * 2)
            + cfg.get("wEvidence", 0) * evidence
            + cfg.get("wTrend", 0) * s["trend"]
        )
        if not math.isfinite(signal):
            continue
        w = signal
        if cfg.get("contrarian") and s["accBayes"] < 0.5:
            w *= -1
        boost = math.pow(max(1, s["n"]), cfg.get("nBoost", 0.5))
        w *= boost
        if abs(w) < 1e-9:
            continue
        weights[slug] = w
    sum_abs = sum(abs(x) for x in weights.values())
    if not sum_abs:
        return {}
    return {slug: w / sum_abs for slug, w in weights.items()}


def predict_with_fusion_config(pred_by_year, outcomes, target, year, cfg, cache):
    key = f"{cfg['id']}:{year}"
    if key in cache:
        return cache[key]
    preds = pred_by_year.get(year, [])
    if not preds:
        empty = {"pred": "", "certainty": float("nan"), "used": 0, "usedWeighted": False}
        cache[key] = empty
        return empty
    stats_res = compute_fusion_stats(pred_by_year, outcomes, target, year, cfg)
    weights = build_fusion_weights(stats_res["stats"], stats_res["maxN"], cfg)
    score = 0.0
    total_abs = 0.0
    used = 0
    for p in preds:
        w = weights.get(p.get("groundhogSlug"))
        if w is None:
            continue
        out = prediction_to_outcome(bool(p.get("shadow")))
        vote = 1 if out == "EARLY_SPRING" else -1
        score += w * vote
        total_abs += abs(w)
        used += 1
    if not total_abs:
        fallback = majority_vote(preds)
        res = {**fallback, "usedWeighted": False}
    else:
        pred = "EARLY_SPRING" if score >= 0 else "LONG_WINTER"
        certainty = min(1.0, abs(score) / total_abs)
        res = {"pred": pred, "certainty": certainty, "used": used, "usedWeighted": True}
    cache[key] = res
    return res


def signal_from_prediction(res: dict):
    if not res or not res.get("pred"):
        return {"signal": 0, "strength": 0}
    sign = 1 if res["pred"] == "EARLY_SPRING" else -1
    certainty = res["certainty"] if math.isfinite(res.get("certainty", float("nan"))) else 0
    strength = sign * (0.35 + 0.65 * certainty)
    return {"signal": sign, "strength": strength}


def build_fusion_feature_cache(pred_by_year, outcomes, target, configs, min_groundhogs):
    all_years = sorted(pred_by_year.keys())
    scored_years = [y for y in all_years if f"{target}:{y}" in outcomes and has_min_groundhogs(pred_by_year, y, min_groundhogs)]
    results_by_year = {}
    pred_cache = {}
    for y in all_years:
        results_by_year[y] = [predict_with_fusion_config(pred_by_year, outcomes, target, y, cfg, pred_cache) for cfg in configs]
    labels_by_year = {}
    for y in scored_years:
        out = outcomes.get(f"{target}:{y}")
        labels_by_year[y] = 1 if out == "EARLY_SPRING" else 0
    return {
        "target": target,
        "configs": configs,
        "allYears": all_years,
        "scoredYears": scored_years,
        "resultsByYear": results_by_year,
        "labelsByYear": labels_by_year,
    }


def config_performance(feature_cache, config_index, train_years, opts):
    k = 0.0
    n = 0.0
    k_early = 0.0
    n_early = 0.0
    k_late = 0.0
    n_late = 0.0
    current_year = opts.get("currentYear", max(train_years))
    lambda_decay = math.log(2) / max(1e-9, opts.get("decayHalfLife", 0)) if opts.get("decayHalfLife") else None
    split_year = train_years[len(train_years) // 2]

    for y in train_years:
        res = feature_cache["resultsByYear"].get(y, [None])[config_index]
        if not res or not res.get("pred"):
            continue
        actual = "EARLY_SPRING" if feature_cache["labelsByYear"].get(y) == 1 else "LONG_WINTER"
        correct = res["pred"] == actual
        age = current_year - y
        weight = math.exp(-lambda_decay * age) if lambda_decay is not None else 1.0
        n += weight
        if correct:
            k += weight
        if y < split_year:
            n_early += weight
            if correct:
                k_early += weight
        else:
            n_late += weight
            if correct:
                k_late += weight
    acc = k / n if n else float("nan")
    acc_early = k_early / n_early if n_early else acc
    acc_late = k_late / n_late if n_late else acc
    stability = clamp(1 - abs(acc_early - acc_late), 0, 1) if math.isfinite(acc_early) and math.isfinite(acc_late) else 0.5
    return {"k": k, "n": n, "acc": acc, "stability": stability}


def select_top_config_indexes(feature_cache, year, cfg):
    window_start = year - cfg.get("rankWindowYears") if cfg.get("rankWindowYears") else None
    train_years = [y for y in feature_cache["scoredYears"] if y < year and (window_start is None or y >= window_start)]
    if not train_years:
        return list(range(len(feature_cache["configs"])))
    scored = []
    for idx, _ in enumerate(feature_cache["configs"]):
        perf = config_performance(
            feature_cache,
            idx,
            train_years,
            {"currentYear": year, "decayHalfLife": cfg.get("rankDecayHalfLife")}
        )
        if not math.isfinite(perf["acc"]) or perf["n"] < cfg.get("minConfigYears", 6):
            continue
        p = (perf["k"] + 2) / (perf["n"] + 4)
        scored.append({"idx": idx, "score": p, "n": perf["n"]})
    if not scored:
        return list(range(len(feature_cache["configs"])))
    scored.sort(key=lambda s: (-s["score"], -s["n"]))
    top_k = max(1, min(cfg.get("topK", len(scored)), len(scored)))
    return [s["idx"] for s in scored[:top_k]]


def build_feature_vector(feature_cache, year, config_idxs):
    results = feature_cache["resultsByYear"].get(year, [])
    feats = []
    used = 0
    sum_signal = 0.0
    sum_strength = 0.0
    sum_abs_strength = 0.0
    for idx in config_idxs:
        res = results[idx] if idx < len(results) else None
        sig = signal_from_prediction(res)
        if sig["signal"] != 0:
            used += 1
        sum_signal += sig["signal"]
        sum_strength += sig["strength"]
        sum_abs_strength += abs(sig["strength"])
        feats.extend([sig["signal"], sig["strength"]])
    used_ratio = used / len(config_idxs) if config_idxs else 0
    mean_signal = sum_signal / used if used else 0
    mean_strength = sum_strength / used if used else 0
    mean_abs_strength = sum_abs_strength / used if used else 0
    consensus = abs(sum_signal) / used if used else 0
    disagreement = 1 - consensus
    feats.extend([mean_signal, mean_strength, mean_abs_strength, consensus, disagreement, used_ratio])
    return feats, used


def train_logistic(features, labels, opts):
    steps = opts.get("steps", 200)
    lr = opts.get("lr", 0.2)
    l2 = opts.get("l2", 0.01)
    if not features:
        return None
    m = len(features[0])
    if not m:
        return None
    w = [0.0] * m
    b = 0.0
    n = len(features)
    sample_weights = opts.get("sampleWeights")
    weight_sum = sum(sample_weights) if sample_weights else n
    inv_n = 1 / weight_sum if weight_sum else 0
    for _ in range(steps):
        grad_w = [0.0] * m
        grad_b = 0.0
        for i in range(n):
            x = features[i]
            wi = sample_weights[i] if sample_weights else 1.0
            z = b + sum(w[j] * x[j] for j in range(m))
            p = 1 / (1 + math.exp(-z))
            diff = (p - labels[i]) * wi
            for j in range(m):
                grad_w[j] += diff * x[j]
            grad_b += diff
        for j in range(m):
            grad_w[j] = grad_w[j] * inv_n + l2 * w[j]
            w[j] -= lr * grad_w[j]
        grad_b *= inv_n
        b -= lr * grad_b
    return {"w": w, "b": b}


def stacked_fusion_predict(feature_cache, year, config_idxs, opts):
    window_start = year - opts.get("windowYears") if opts.get("windowYears") else None
    train_years = [y for y in feature_cache["scoredYears"] if y < year and (window_start is None or y >= window_start)]
    if len(train_years) < opts.get("minTrain", 12):
        return {"pred": "", "certainty": float("nan"), "used": 0}
    X = []
    y = []
    sample_weights = []
    lambda_decay = math.log(2) / max(1e-9, opts.get("decayHalfLife", 0)) if opts.get("decayHalfLife") else None
    for train_year in train_years:
        feats, used = build_feature_vector(feature_cache, train_year, config_idxs)
        if not used:
            continue
        X.append(feats)
        y.append(feature_cache["labelsByYear"].get(train_year))
        if lambda_decay is not None:
            sample_weights.append(math.exp(-lambda_decay * (year - train_year)))
    if len(X) < opts.get("minTrain", 12):
        return {"pred": "", "certainty": float("nan"), "used": 0}
    model = train_logistic(X, y, {**opts, "sampleWeights": sample_weights if lambda_decay is not None else None})
    if not model:
        return {"pred": "", "certainty": float("nan"), "used": 0}
    feats, used = build_feature_vector(feature_cache, year, config_idxs)
    if not used:
        return {"pred": "", "certainty": float("nan"), "used": 0}
    z = model["b"] + sum(model["w"][j] * feats[j] for j in range(len(model["w"])))
    p = 1 / (1 + math.exp(-z))
    pred = "EARLY_SPRING" if p >= 0.5 else "LONG_WINTER"
    certainty = abs(p - 0.5) * 2
    return {"pred": pred, "certainty": certainty, "used": used}


def weighted_blend_predict(feature_cache, year, config_idxs, opts):
    window_start = year - opts.get("windowYears") if opts.get("windowYears") else None
    train_years = [y for y in feature_cache["scoredYears"] if y < year and (window_start is None or y >= window_start)]
    if len(train_years) < opts.get("minTrain", 8):
        return {"pred": "", "certainty": float("nan"), "used": 0}
    score = 0.0
    total_abs = 0.0
    used = 0
    for idx in config_idxs:
        perf = config_performance(feature_cache, idx, train_years, {
            "currentYear": year,
            "decayHalfLife": opts.get("decayHalfLife"),
        })
        if perf["n"] < opts.get("minConfigYears", 6):
            continue
        p = (perf["k"] + 2) / (perf["n"] + 4)
        stability_factor = 1 + opts.get("stabilityBoost", 0) * (perf["stability"] - 0.5)
        weight = logit(p) * (1 + math.log1p(perf["n"]) / 3) * stability_factor
        res = feature_cache["resultsByYear"].get(year, [None])[idx]
        sig = signal_from_prediction(res)
        if not sig["signal"]:
            continue
        score += weight * sig["strength"]
        total_abs += abs(weight)
        used += 1
    if not total_abs:
        return {"pred": "", "certainty": float("nan"), "used": 0}
    pred = "EARLY_SPRING" if score >= 0 else "LONG_WINTER"
    certainty = min(1.0, abs(score) / total_abs)
    return {"pred": pred, "certainty": certainty, "used": used}


def dynamic_super_predict(pred_by_year, year, model):
    config_idxs = select_top_config_indexes(model["featureCache"], year, model)
    res = stacked_fusion_predict(model["featureCache"], year, config_idxs, model["stack"])
    method = "stacked"
    used_ratio = res["used"] / len(config_idxs) if config_idxs else 0
    if not res["pred"] or res["certainty"] < model["gate"] or res["used"] < model["minModels"] or used_ratio < model.get("minUsedRatio", 0):
        blended = weighted_blend_predict(model["featureCache"], year, config_idxs, {
            **model["blend"],
            "minConfigYears": model["minConfigYears"],
        })
        if blended["pred"]:
            res = blended
            method = "blend"
    if not res["pred"]:
        res = majority_vote(pred_by_year.get(year, []))
        method = "majority"
    return {**res, "method": method}


def backtest_dynamic_super(pred_by_year, outcomes, model):
    k = 0
    n = 0
    last_year = None
    for y in model["featureCache"]["scoredYears"]:
        res = dynamic_super_predict(pred_by_year, y, model)
        if not res["pred"]:
            continue
        n += 1
        if res["pred"] == outcomes.get(f"{model['target']}:{y}"):
            k += 1
        last_year = y
    return {"accuracy": k / n if n else float("nan"), "backtestN": n, "lastYear": last_year}


def tune_dynamic_super(pred_by_year, outcomes, feature_cache, tuning_sets):
    best = None
    for tuning in tuning_sets:
        candidate = {**tuning, "target": feature_cache["target"], "configs": feature_cache["configs"], "featureCache": feature_cache}
        backtest = backtest_dynamic_super(pred_by_year, outcomes, candidate)
        if not math.isfinite(backtest["accuracy"]):
            continue
        candidate["backtest"] = backtest
        if best is None or backtest["accuracy"] > best["backtest"]["accuracy"] or (
            backtest["accuracy"] == best["backtest"]["accuracy"] and backtest["backtestN"] > best["backtest"]["backtestN"]
        ):
            best = candidate
    return best


def build_dynamic_super_model(pred_by_year, outcomes, target, configs, tuning_sets):
    feature_cache = build_fusion_feature_cache(pred_by_year, outcomes, target, configs, MIN_BACKTEST_GH)
    tuned = tune_dynamic_super(pred_by_year, outcomes, feature_cache, tuning_sets)
    return tuned


def evaluate(target: str):
    pred_text = Path("docs/data/predictions.json").read_text()
    outcomes_text = Path("docs/data/outcomes.csv").read_text()
    pred_obj = json.loads(pred_text)
    outcomes_rows = parse_csv(outcomes_text)
    outcomes = index_outcomes(build_outcome_rows(outcomes_rows))
    pred_by_year = index_predictions(pred_obj)
    model = build_dynamic_super_model(pred_by_year, outcomes, target, FUSION_CONFIGS, TUNING_SETS)
    if not model:
        return None
    return model


def main():
    model = evaluate(TARGET_BASE)
    if not model:
        print("Dynamic super model could not be evaluated.")
        return
    acc = model["backtest"]["accuracy"]
    print(f"Target: {model['target']}")
    print(f"Fusion profile: {model['id']}")
    print(f"Backtest accuracy: {acc * 100:.1f}% (goal {GOAL_ACCURACY * 100:.0f}%: {'reached' if acc >= GOAL_ACCURACY else 'not reached'})")
    print(f"Backtest years: {model['backtest']['backtestN']}")
    print(f"Last scored year: {model['backtest']['lastYear']}")


if __name__ == "__main__":
    main()
