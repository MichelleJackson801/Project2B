import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Heuristics to exclude non-rating columns by name ---
EXCLUDE_NAME_PATTERNS = [
    r"timestamp", r"time", r"date",
    r"email", r"name",
    r"comment", r"open\s*ended", r"why", r"explain", r"feedback", r"suggest",
    r"id$", r"uid", r"student",
    r"demographic", r"gender", r"age",
]

def looks_excluded(col: str) -> bool:
    c = col.strip().lower()
    return any(re.search(p, c) for p in EXCLUDE_NAME_PATTERNS)

def infer_rating_columns(df: pd.DataFrame, min_nonnull: int = 10):
    """
    Pick columns that look like Likert ratings:
    - Mostly numeric after coercion
    - Reasonable bounded scale (1–5, 1–7, 0–10, etc.)
    - Enough responses
    """
    candidates = []
    for col in df.columns:
        if looks_excluded(col):
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        nonnull = s.notna().sum()
        if nonnull < min_nonnull:
            continue

        # Must be "mostly numeric" among non-null entries
        # (after coercion we only see numeric)
        unique_vals = s.dropna().unique()
        if len(unique_vals) < 2:
            continue

        vmin, vmax = float(np.nanmin(s)), float(np.nanmax(s))

        # Common rating bounds (allow a bit of slack)
        plausible = (
            (0 <= vmin and vmax <= 10) or
            (1 <= vmin and vmax <= 7) or
            (1 <= vmin and vmax <= 5) or
            (1 <= vmin and vmax <= 9)
        )
        if not plausible:
            continue

        candidates.append(col)

    return candidates

def rank_items_wide(df: pd.DataFrame, rating_cols: list[str]) -> pd.DataFrame:
    """
    Treat each rating column as an "item" (program/course) and compute:
    - mean rating
    - count
    - std dev
    Then rank by mean desc (ties broken by count desc).
    """
    rows = []
    for col in rating_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.dropna()
        rows.append({
            "item": col,
            "mean_rating": s.mean() if len(s) else np.nan,
            "n_responses": int(len(s)),
            "std_dev": s.std(ddof=1) if len(s) > 1 else np.nan,
        })

    out = pd.DataFrame(rows).dropna(subset=["mean_rating"])
    out = out.sort_values(["mean_rating", "n_responses"], ascending=[False, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "item", "mean_rating", "n_responses", "std_dev"]]

def rank_items_long(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    If dataset is already long-form with obvious columns like:
      program/course/item + rating/score
    we can detect and use that instead of wide-form inference.
    """
    cols = {c.lower().strip(): c for c in df.columns}
    possible_item = None
    possible_rating = None

    for k in cols:
        if k in ["program", "course", "item", "class"]:
            possible_item = cols[k]
        if k in ["rating", "score", "preference"]:
            possible_rating = cols[k]

    if not possible_item or not possible_rating:
        return None

    temp = df[[possible_item, possible_rating]].copy()
    temp[possible_rating] = pd.to_numeric(temp[possible_rating], errors="coerce")
    temp = temp.dropna(subset=[possible_item, possible_rating])

    grouped = temp.groupby(possible_item)[possible_rating].agg(["mean", "count", "std"]).reset_index()
    grouped = grouped.rename(columns={
        possible_item: "item",
        "mean": "mean_rating",
        "count": "n_responses",
        "std": "std_dev",
    })
    grouped = grouped.sort_values(["mean_rating", "n_responses"], ascending=[False, False]).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped[["rank", "item", "mean_rating", "n_responses", "std_dev"]]

def plot_rankings(ranks: pd.DataFrame, outpath: Path, top_n: int = 20):
    top = ranks.head(top_n).copy()
    top = top.sort_values("mean_rating", ascending=True)  # for horizontal bar chart

    plt.figure(figsize=(10, max(6, 0.35 * len(top))))
    plt.barh(top["item"], top["mean_rating"])
    plt.xlabel("Average Rating")
    plt.title(f"Rank-Ordered Programs/Courses (Top {min(top_n, len(ranks))})")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = os.getenv("DATA_PATH", str(repo_root / "data" / "exit_survey_202X.csv"))
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

if str(data_path).lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(data_path)
else:
    if str(data_path).lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(data_path)
else:
    df = pd.read_csv(data_path)
    
    # First try long-form detection
    ranks = rank_items_long(df)

    # If not long-form, infer rating columns in wide format
    if ranks is None:
        rating_cols = infer_rating_columns(df, min_nonnull=int(os.getenv("MIN_NONNULL", "10")))
        if not rating_cols:
            raise ValueError(
                "No rating columns detected. "
                "Rename columns to include Program/Course + Rating, OR ensure Likert columns are numeric-like."
            )
        ranks = rank_items_wide(df, rating_cols)

    # Save table
    ranks.to_csv(outputs_dir / "rank_table.csv", index=False)

    # Save figure
    top_n = int(os.getenv("TOP_N", "20"))
    plot_rankings(ranks, outputs_dir / "rank_order.png", top_n=top_n)

    # Optional: create a reflection template (YOU must edit this yourself if you use it)
    reflection_path = outputs_dir / "reflection.md"
    if not reflection_path.exists():
        reflection_path.write_text(
            "# Reflection (write this in your own words)\n\n"
            "## What changed from Project 1 to this workflow?\n- \n\n"
            "## Where is the control now?\n- \n\n"
            "## What would you do next if you had one more week?\n- \n\n"
            "## One accounting application of this workflow (be specific)\n- \n",
            encoding="utf-8"
        )

    print("Done. Outputs written to:", outputs_dir)

if __name__ == "__main__":
    main()
