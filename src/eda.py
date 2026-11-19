
"""Focused Home Credit organization and employment analysis."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "input" / "train.csv"
OUT = ROOT / "outputs" / "eda"


def organization_xna_analysis(train: pd.DataFrame) -> None:
    """Analyze XNA and non-XNA organization groups."""
    if "ORGANIZATION_TYPE" not in train.columns:
        print("ORGANIZATION_TYPE not found; skipping XNA analysis.")
        return

    data = train[["ORGANIZATION_TYPE", "TARGET"]].copy()

    org = (
        data["ORGANIZATION_TYPE"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    data["ORGANIZATION_XNA"] = org.eq("XNA").map({
        True: "XNA",
        False: "Non-XNA",
    })

    summary = (
        data.groupby("ORGANIZATION_XNA", dropna=False)
        .agg(
            applications=("TARGET", "size"),
            defaults=("TARGET", "sum"),
            default_rate=("TARGET", "mean"),
        )
        .reset_index()
    )

    summary["application_share"] = (
        summary["applications"] / len(data)
    )

    summary.to_csv(
        OUT / "organization_xna_summary.csv",
        index=False,
    )

    print("\nORGANIZATION_TYPE XNA ANALYSIS")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(
        summary["ORGANIZATION_XNA"],
        summary["applications"],
    )
    axes[0].set_title("Application count")
    axes[0].set_ylabel("Applications")

    axes[1].bar(
        summary["ORGANIZATION_XNA"],
        summary["default_rate"] * 100,
    )
    axes[1].set_title("Observed default rate")
    axes[1].set_ylabel("TARGET = 1 (%)")

    fig.tight_layout()
    fig.savefig(
        OUT / "organization_xna_analysis.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def employment_sentinel_analysis(train: pd.DataFrame) -> None:
    """Summarize the positive DAYS_EMPLOYED sentinel-like values."""
    if "DAYS_EMPLOYED" not in train.columns:
        print("DAYS_EMPLOYED not found; skipping sentinel analysis.")
        return

    data = train[["DAYS_EMPLOYED", "TARGET"]].copy()

    days = pd.to_numeric(
        data["DAYS_EMPLOYED"], errors="coerce"
    )

    data["EMPLOYMENT_GROUP"] = "Negative duration"
    data.loc[days > 0, "EMPLOYMENT_GROUP"] = "Positive sentinel-like"
    data.loc[days == 0, "EMPLOYMENT_GROUP"] = "Zero"
    data.loc[days.isna(), "EMPLOYMENT_GROUP"] = "Missing"

    summary = (
        data.groupby("EMPLOYMENT_GROUP", dropna=False)
        .agg(
            applications=("TARGET", "size"),
            defaults=("TARGET", "sum"),
            default_rate=("TARGET", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT / "employment_sentinel_summary.csv",
        index=False,
    )

    print("\nEMPLOYMENT SENTINEL ANALYSIS")
    print(summary.to_string(index=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)

    organization_xna_analysis(train)
    employment_sentinel_analysis(train)

    print(f"\nEDA outputs saved in: {OUT}")


if __name__ == "__main__":
    main()
