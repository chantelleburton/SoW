import os
import re
import glob
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Input/output locations
BASELINE_DIR = "/data/scratch/bob.potts/sowf/test_output/Baseline/Sub-Indicies"
PLOT_DIR = "/data/scratch/bob.potts/sowf/test_output/Plots/Sub-Indicies/Timeseries"

# Match both 1980-2013 and 1960-2013 variants
FILE_GLOB = os.path.join(BASELINE_DIR, "ERA5_*_*_*_95%.csv")

# Optional fixed display order
COUNTRY_ORDER = ["Korea", "Iberia", "Scotland", "Chile", "Canada"]
INDEX_ORDER = ["FWI", "FFMC", "DMC", "DC", "ISI", "BUI"]


def parse_metadata_from_filename(path):
    """
    Expected: ERA5_<INDEX>_<YYYY-YYYY>_<COUNTRY>_95%.csv
    """
    fname = os.path.basename(path)
    m = re.match(r"^ERA5_([A-Za-z0-9]+)_([0-9]{4}-[0-9]{4})_(.+)_95%\.csv$", fname)
    if not m:
        return None
    idx, period, country = m.group(1), m.group(2), m.group(3)
    return {"index": idx, "period": period, "country": country, "filename": fname}


def parse_year(date_value):
    """
    Handles Date values such as:
    - 1980-03
    - 1980-06/07
    - 1980-07/08
    """
    if pd.isna(date_value):
        return None
    s = str(date_value).strip()
    m = re.match(r"^([0-9]{4})-", s)
    if not m:
        return None
    return int(m.group(1))


def parse_month_window(date_value):
    """
    Returns month token for labeling, e.g.:
    - 03
    - 06/07
    - 07/08
    """
    if pd.isna(date_value):
        return ""
    s = str(date_value).strip()
    m = re.match(r"^[0-9]{4}-(.+)$", s)
    if not m:
        return ""
    return m.group(1)


def build_label_from_month_token(token):
    if not token:
        return ""
    if "/" in token:
        parts = token.split("/")
        return f"months {'-'.join(parts)}"
    return f"month {token}"


def plot_single_series(csv_path, out_dir):
    meta = parse_metadata_from_filename(csv_path)
    if meta is None:
        print(f"Skipping unexpected filename format: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "Date" not in df.columns or "FWI" not in df.columns:
        print(f"Skipping {csv_path}: expected columns Date and FWI not found")
        return

    df["Year"] = df["Date"].apply(parse_year)
    df = df.dropna(subset=["Year", "FWI"]).copy()
    if df.empty:
        print(f"Skipping {csv_path}: no valid data rows")
        return

    df["Year"] = df["Year"].astype(int)
    df = df.sort_values("Year")

    # Derive month window label from first row (all rows in file should share same pattern)
    month_token = parse_month_window(df["Date"].iloc[0])
    month_label = build_label_from_month_token(month_token)

    country = meta["country"]
    idx = meta["index"]
    period = meta["period"]

    os.makedirs(out_dir, exist_ok=True)
    out_name = f"ERA5_{idx}_{country}_{period}_95pct_timeseries.png"
    out_path = os.path.join(out_dir, out_name)

    plt.figure(figsize=(10, 5))
    plt.plot(df["Year"], df["FWI"], marker="o", linewidth=1.6, markersize=3.5)
    plt.title(f"ERA5 {idx} 95th Percentile Baseline - {country} ({period})")
    plt.xlabel("Year")
    plt.ylabel(f"{idx} value")
    if month_label:
        plt.suptitle(month_label, y=0.94, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")


def sort_key(path):
    meta = parse_metadata_from_filename(path)
    if meta is None:
        return ("ZZZ", "ZZZ", "9999-9999", os.path.basename(path))
    country_rank = COUNTRY_ORDER.index(meta["country"]) if meta["country"] in COUNTRY_ORDER else 999
    index_rank = INDEX_ORDER.index(meta["index"]) if meta["index"] in INDEX_ORDER else 999
    return (country_rank, index_rank, meta["period"], meta["filename"])


def main():
    csv_files = sorted(glob.glob(FILE_GLOB), key=sort_key)
    if not csv_files:
        print(f"No CSV files found with: {FILE_GLOB}")
        return

    print(f"Found {len(csv_files)} CSV files")
    for csv_path in csv_files:
        plot_single_series(csv_path, PLOT_DIR)


if __name__ == "__main__":
    main()