"""
VaxAlert - Tool 1: Data Analysis Tool
Loads vaccination dataset, computes coverage stats,
and identifies low-coverage countries.
"""

import pandas as pd
import numpy as np
import os


def load_vaccination_data(filepath: str = "data/vaccinations.csv") -> pd.DataFrame:
    """
    Load the COVID-19 World Vaccination Progress dataset.
    Expected columns: country, date, people_vaccinated_per_hundred, daily_vaccinations
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            "Download from: https://www.kaggle.com/datasets/gpreda/covid-world-vaccination-progress\n"
            "Place the file at: data/vaccinations.csv"
        )

    df = pd.read_csv(filepath, parse_dates=["date"])

    # Keep only relevant columns
    cols_needed = ["country", "date", "people_vaccinated_per_hundred", "daily_vaccinations"]
    df = df[[c for c in cols_needed if c in df.columns]].copy()

    # Fill missing values with 0 for numeric columns
    df["people_vaccinated_per_hundred"] = df["people_vaccinated_per_hundred"].fillna(0)
    df["daily_vaccinations"] = df["daily_vaccinations"].fillna(0)

    return df


def get_latest_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the most recent vaccination coverage per country.
    Returns a DataFrame sorted by coverage (ascending = lowest first).
    """
    # Get latest date per country
    latest = (
        df.sort_values("date")
        .groupby("country")
        .last()
        .reset_index()
    )

    # Filter out countries with 0 or missing coverage (no data)
    latest = latest[latest["people_vaccinated_per_hundred"] > 0]

    # Sort ascending — lowest coverage first
    latest = latest.sort_values("people_vaccinated_per_hundred", ascending=True)

    return latest[["country", "date", "people_vaccinated_per_hundred"]]


def identify_low_coverage_countries(filepath: str = "data/vaccinations.csv", top_n: int = 10) -> dict:
    """
    VaxAlert Data Analysis Tool.
    Identifies the top_n countries with lowest vaccination coverage.

    Args:
        filepath: Path to the vaccination CSV dataset
        top_n: Number of low-coverage countries to return (default: 10)

    Returns:
        dict with keys:
            - low_coverage_countries: list of {country, coverage_rate, last_updated}
            - global_average: float
            - analysis_summary: str
    """
    try:
        df = load_vaccination_data(filepath)
        latest = get_latest_coverage(df)

        # Global average
        global_avg = round(latest["people_vaccinated_per_hundred"].mean(), 2)

        # Bottom N countries
        bottom_n = latest.head(top_n)

        low_coverage_list = []
        for _, row in bottom_n.iterrows():
            low_coverage_list.append({
                "country": row["country"],
                "coverage_rate": round(row["people_vaccinated_per_hundred"], 2),
                "last_updated": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])
            })

        # Build summary text for agent context
        summary_lines = [
            f"Global average vaccination coverage: {global_avg}%",
            f"Top {top_n} countries with lowest coverage:"
        ]
        for item in low_coverage_list:
            summary_lines.append(
                f"  - {item['country']}: {item['coverage_rate']}% (as of {item['last_updated']})"
            )

        return {
            "low_coverage_countries": low_coverage_list,
            "global_average": global_avg,
            "analysis_summary": "\n".join(summary_lines)
        }

    except FileNotFoundError as e:
        return {
            "error": str(e),
            "low_coverage_countries": [],
            "global_average": None,
            "analysis_summary": "Dataset not found. Please download it first."
        }
