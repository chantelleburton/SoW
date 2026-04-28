#this file is used to create random offsets in th existing bias correction to test the multi year functionality of 
# Plotting/Explore_Risk_Ratio.py. 

import os
import re
import glob
import pandas as pd
import numpy as np

# ---- Configuration ----
input_dir = "/data/scratch/bob.potts/sowf/test_output/Testing_Log_Transforms"
output_dir = "/data/scratch/bob.potts/sowf/test_output/Testing_Log_Transforms_Modified"
new_data_year = "2025"  # Change this to whatever DataYear you want
# ------------------------

os.makedirs(output_dir, exist_ok=True)

csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
print(f"Found {len(csv_files)} CSV files")

for filepath in csv_files:
    df = pd.read_csv(filepath)

    # Apply random perturbation of up to ±10% to all columns except Year
    value_cols = [c for c in df.columns if c != "Year"]
    random_factors = np.random.uniform(-0.1, 0.1, size=(len(df), len(value_cols)))
    df[value_cols] = df[value_cols].values * (1 + random_factors)

    # Replace DataYear in the filename
    original_name = os.path.basename(filepath)
    new_name = re.sub(r"DataYear_\d+", f"DataYear_{new_data_year}", original_name)

    output_path = os.path.join(output_dir, new_name)
    df.to_csv(output_path, index=False)

print(f"Done — {len(csv_files)} files written to {output_dir}")