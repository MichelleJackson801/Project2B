import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CORE_RANK_PREFIX = "Please place each MAcc CORE course into rank order"
ELECTIVE_RATE_PREFIX = "Rate ACC"


def clean_course_name(col: str) -> str:
    # Keep the part after the last " - "
    parts = str(col).split(" - ")
    return parts[-1].strip() if len(parts) > 1 else str(col).strip()


def find_core_rank_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if CORE_RANK_PREFIX.lower() in str(c).lower()]


def find_elective_rate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).strip().startswith(ELECTIVE_RATE_PREFIX)]


def rank_core_courses(df: pd.DataFrame) -> pd.DataFrame:
    cols = find_core_rank_cols(df)
    if not cols:
        raise ValueError(
            "No CORE rank-order columns found. Expected columns containing: "
            "'Please place each MAcc CORE course into rank order'"
        )

    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        # CORE ranks are typically 1..8; keep a wide safety bound
        s = s[(s >= 1) & (s <= 50)]
        if len(s) == 0:
            continue
        rows.append({
            "course": clean_course_name(col),
            "mean_rank": float(s.mean()),
            "n_responses": int(len(s)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("CORE rank columns found, but no usable numeric rank values.")

    # Lower mean rank = better (more beneficial)
    out = out.sort_values(["mean_rank", "n_responses"], ascending=[True, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "course", "mean_rank", "n_responses"]]


def rank_electives(df: pd.DataFrame) -> pd.DataFrame:
    cols = find_elective_rate_cols(df)
    if not cols:
        return pd.DataFrame(columns=["rank", "course", "mean_rating", "n_responses"])

    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        s = s[(s >= 1) & (s <= 5)]
        if len(s) == 0:
            continue
        rows.append({
            "course": clean_course_name(col),
            "mean_rating": float(s.mean()),
            "n_responses": int(len(s)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["rank", "course", "mean_rating", "n_responses"])

    # Higher mean rating = better
    out = out.sort_values(["mean_rating", "n_responses"], ascending=[False, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "course", "mean_rating", "n_responses"]]


def plot_core_rank(core_table: pd.DataFrame, outpath: Path, top_n: int = 20) -> None:
    top = core_table.head(top_n).copy()
    # For horizontal bar chart, reverse so best appears on top
    top = top.sort_values("mean_rank", ascending=False)

    plt.figure(figsize=(10, max(6, 0.35 * len(top))))
    plt.barh(top["course"], top["mean_rank"])
    plt.xlabel("Average Rank (lower = more beneficial)")
    plt.title(f"MAcc CORE Courses: Rank Order (Top {min(top_n, len(core_table))})")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = os.getenv("DATA_PATH", str(repo_root / "data" / "Grad Program Exit Survey Data 2024.xlsx"))
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Read Excel or CSV
    if str(data_path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    core = rank_core_courses(df)
    electives = rank_electives(df)

    core.to_csv(outputs_dir / "rank_table.csv", index=False)
    electives.to_csv(outputs_dir / "elective_rank_table.csv", index=False)

    top_n = int(os.getenv("TOP_N", "20"))
    plot_core_rank(core, outputs_dir / "rank_order.png", top_n=top_n)

    print("Saved:", outputs_dir / "rank_table.csv")
    print("Saved:", outputs_dir / "rank_order.png")
    print("Saved:", outputs_dir / "elective_rank_table.csv")


if __name__ == "__main__":
    main()
