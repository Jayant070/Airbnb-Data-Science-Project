import numpy as np
import pandas as pd


def analyze_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Return IQR-based outlier insights for non-binary numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    non_binary_numeric_cols = []
    for col in numeric_cols:
        vals = df[col].dropna().unique()
        if not set(vals).issubset({0, 1}):
            non_binary_numeric_cols.append(col)

    outlier_summary = []

    for col in non_binary_numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_count = 0
            lower = q1
            upper = q3
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()

        non_na_count = df[col].notna().sum()
        outlier_pct = round((outlier_count / non_na_count) * 100, 2) if non_na_count > 0 else 0.0

        outlier_summary.append(
            {
                "feature": col,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(outlier_count),
                "outlier_pct": outlier_pct,
            }
        )

    if not outlier_summary:
        return pd.DataFrame(columns=["feature", "lower_bound", "upper_bound", "outlier_count", "outlier_pct"])

    return pd.DataFrame(outlier_summary).sort_values("outlier_count", ascending=False).reset_index(drop=True)
