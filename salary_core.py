"""Feature building and prediction logic for the salary app.

Mirrors Section 7 (Feature engineering) of job_postings_fixed.ipynb, so a single
form input becomes exactly the same feature row the model was trained on.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SENIORITY_ORDER = ["junior", "mid", "senior", "staff", "lead"]
SENIORITY_PREFIX = r"^(junior|senior|lead|staff|principal|head of)\s+"


# ---------------------------------------------------------------- reference data
@dataclass
class Reference:
    roles: list[str]                 # display names, e.g. "Data Analyst"
    city_country: dict[str, str]     # "London" -> "GB"
    skills: list[str]                # all skills, most frequent first
    role_skills: dict[str, list[str]]  # role -> its skills, most frequent first
    remote_options: list[str]
    company_sizes: list[str]
    salaries: pd.DataFrame           # labelled postings: role, seniority, city, country, salary_mid_usd


def load_reference(csv_path: str | Path) -> Reference:
    raw = pd.read_csv(csv_path)
    for col in ["title", "seniority", "city", "country", "remote", "company_size", "skills"]:
        raw[col] = raw[col].astype(str).str.strip()

    raw["role"] = raw["title"].str.replace(SENIORITY_PREFIX, "", regex=True, flags=re.I).str.strip()
    raw["skills_list"] = raw["skills"].apply(
        lambda v: sorted({s.strip().lower() for s in str(v).split("|") if s.strip()})
    )

    skill_counts = Counter(s for xs in raw["skills_list"] for s in xs)
    role_skills = {
        role: [s for s, _ in Counter(s for xs in grp["skills_list"] for s in xs).most_common()]
        for role, grp in raw.groupby("role")
    }

    salaries = raw.dropna(subset=["salary_min_usd", "salary_max_usd"]).copy()
    salaries["salary_mid_usd"] = (salaries["salary_min_usd"] + salaries["salary_max_usd"]) / 2

    return Reference(
        roles=raw["role"].value_counts().index.tolist(),
        city_country=raw.groupby("city")["country"].agg(lambda s: s.mode().iloc[0]).to_dict(),
        skills=[s for s, _ in skill_counts.most_common()],
        role_skills=role_skills,
        remote_options=raw["remote"].value_counts().index.tolist(),
        company_sizes=raw["company_size"].value_counts().index.tolist(),
        salaries=salaries[["role", "seniority", "city", "country", "salary_mid_usd"]],
    )


# ---------------------------------------------------------------- features
def make_title(role: str, seniority: str) -> str:
    """'mid' postings have no level word in the title (e.g. 'Data Analyst')."""
    return role if seniority == "mid" else f"{seniority.capitalize()} {role}"


def skill_col_name(skill: str) -> str:
    return "skill_" + re.sub(r"[^a-z0-9]+", "_", skill.lower()).strip("_")


def build_features(
    role: str,
    seniority: str,
    city: str,
    country: str,
    remote: str,
    company_size: str,
    skills: list[str],
    posted_on: date,
    all_skills: list[str],
) -> pd.DataFrame:
    title_clean = make_title(role, seniority).lower().strip()
    skills_norm = sorted({s.strip().lower() for s in skills if s.strip()})
    ts = pd.Timestamp(posted_on)

    row = {
        "title_clean": title_clean,
        "skills_text": " ".join(skills_norm),
        "title_family": re.sub(r"\s+", "_", re.sub(SENIORITY_PREFIX, "", title_clean).strip()),
        "seniority": seniority,
        "country": country,
        "city": city,
        "remote": remote,
        "company_size": company_size,
        "skills_count": len(skills_norm),
        "posted_month": ts.month,
        "posted_quarter": ts.quarter,
        "title_word_count": len(title_clean.split()),
    }
    for skill in all_skills:
        row[skill_col_name(skill)] = int(skill in skills_norm)
    return pd.DataFrame([row])


# ---------------------------------------------------------------- model
def load_model(path: str | Path):
    return joblib.load(path)


def model_input_columns(model) -> list[str] | None:
    """Columns the fitted model expects (sklearn pipelines / CatBoost)."""
    for attr in ("feature_names_in_", "feature_names_"):
        cols = getattr(model, attr, None)
        if cols is not None and len(cols):
            return list(cols)
    return None


def predict(model, features: pd.DataFrame) -> np.ndarray:
    cols = model_input_columns(model)
    X = features.copy()
    if cols is not None:
        for c in cols:  # any skill flag the app doesn't know about -> 0
            if c not in X.columns:
                X[c] = 0
        X = X[cols]
    return np.asarray(model.predict(X), dtype=float)


def predict_variants(model, base: dict, field: str, values: list, all_skills: list[str],
                     city_country: dict[str, str] | None = None) -> pd.DataFrame:
    """Predict the same profile while varying one field (e.g. every seniority level or every city)."""
    rows = []
    for v in values:
        params = {**base, field: v}
        if field == "city" and city_country:
            params["country"] = city_country[v]
        rows.append(build_features(**params, all_skills=all_skills))
    preds = predict(model, pd.concat(rows, ignore_index=True))
    return pd.DataFrame({field: values, "predicted_salary_usd": preds})


def model_mape(comparison_csv: str | Path, default: float = 5.0) -> tuple[str | None, float]:
    """Test-set MAPE of the best non-baseline model from salary_model_comparison.csv."""
    try:
        res = pd.read_csv(comparison_csv)
        res = res[~res["Model"].str.startswith("Baseline")].sort_values("MAE")
        return res.iloc[0]["Model"], float(res.iloc[0]["MAPE_%"])
    except Exception:
        return None, default
