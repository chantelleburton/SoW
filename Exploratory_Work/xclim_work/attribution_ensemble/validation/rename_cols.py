#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import pandas as pd

ENS_RE = re.compile(r"^Ens(\d+)_Real(\d+)$")

def convert_col(name: str) -> str:
    m = ENS_RE.match(name)
    if not m:
        return name
    ens = int(m.group(1))
    real = int(m.group(2))
    return f"r{ens:03d}i1p{real}"

def convert_file(csv_path: Path, out_path: Path, overwrite: bool = False) -> None:
    df = pd.read_csv(csv_path)
    old_cols = list(df.columns)
    new_cols = [convert_col(c) for c in old_cols]

    changed = sum(o != n for o, n in zip(old_cols, new_cols))
    if changed == 0:
        print(f"[skip] {csv_path.name}: no EnsX_RealY columns found")
        return

    # Guard against duplicate columns after rename
    if len(set(new_cols)) != len(new_cols):
        raise ValueError(f"Duplicate column names after conversion in {csv_path}")

    df.columns = new_cols

    if overwrite:
        out_path = csv_path
    df.to_csv(out_path, index=False)
    print(f"[ok] {csv_path.name} -> {out_path.name} ({changed} columns renamed)")

def main():
    p = argparse.ArgumentParser(
        description="Rename EnsX_RealY columns to rXXXi1pY in xclim condensed log-transform CSVs."
    )
    p.add_argument(
        "input",
        help="CSV file path or directory containing CSVs"
    )
    p.add_argument(
        "--glob",
        default="*.csv",
        help="File glob when input is a directory (default: *.csv)"
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input files instead of writing *_renamed.csv"
    )
    args = p.parse_args()

    inp = Path(args.input)

    if inp.is_file():
        out = inp if args.overwrite else inp.with_name(inp.stem + "_renamed.csv")
        convert_file(inp, out, overwrite=args.overwrite)
    elif inp.is_dir():
        files = sorted(inp.glob(args.glob))
        if not files:
            print(f"No files matched: {inp / args.glob}")
            return
        for f in files:
            out = f if args.overwrite else f.with_name(f.stem + "_renamed.csv")
            convert_file(f, out, overwrite=args.overwrite)
    else:
        raise FileNotFoundError(inp)

if __name__ == "__main__":
    main()