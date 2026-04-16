import iris
import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", module="iris")
# Define months for each country
country_months = {
    'Korea': [3],
    'Iberia': [8],
    'Scotland': [6, 7],
    'Chile': [1, 2],
    'Canada': [7, 8],
}

countries = ['Korea', 'Iberia', 'Scotland', 'Chile', 'Canada']
indices = ['FWI', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']

event_dir = '/data/scratch/bob.potts/sowf/test_output/Exports/ContinuityCubes'
hist_dir = '/data/scratch/bob.potts/sowf/test_output/Baseline/Sub-Indicies'
output_csv = '/data/scratch/bob.potts/sowf/test_output/Exports/summary_95th_percentiles.csv'

event_results = {}
hist_results = {}
rows = []

for country in countries:
    event_results[country] = {}
    hist_results[country] = {}
    months = country_months[country]
    for idx in indices:
        # Event percentile from .nc
        nc_path = f"{event_dir}/{idx}_{country}_2023-2026.nc"
        event_val = np.nan
        if os.path.exists(nc_path):
            try:
                cube = iris.load_cube(nc_path)
                def event_month(cell):
                    dt = cell.point
                    if country == 'Chile':
                        return (dt.year == 2026) and (dt.month in months)
                    else:
                        return (dt.year == 2025) and (dt.month in months)
                cube_sel = cube.extract(iris.Constraint(time=event_month))
                if cube_sel is not None and cube_sel.shape[0] > 0:
                    event_val = cube_sel.collapsed(['time', 'latitude', 'longitude'], iris.analysis.PERCENTILE, percent=95).data.item()
            except Exception as e:
                print(f"Failed event extraction for {idx} {country}: {e}")
        event_results[country][idx] = event_val

        # Historical average from CSV
        hist_val = np.nan
        csv_path = f"{hist_dir}/ERA5_{idx}_1980-2013_{country}_95%.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df['month'] = df['Date'].str[-2:].astype(int)
                df_sel = df[df['month'].isin(months)]
                if not df_sel.empty:
                    hist_val = df_sel['FWI'].mean()
            except Exception as e:
                print(f"Failed hist extraction for {idx} {country}: {e}")
        hist_results[country][idx] = hist_val

        rows.append({
            'Country': country,
            'Index': idx,
            'Event_95th': event_val,
            'HistAvg_95th': hist_val
        })

# Output summary table
print(f"{'Country':<10} {'Index':<6} {'Event_95th':>12} {'HistAvg_95th':>14}")
print('-'*45)
for row in rows:
    print(f"{row['Country']:<10} {row['Index']:<6} {row['Event_95th']:12.2f} {row['HistAvg_95th']:14.2f}")

# Output CSV
df_out = pd.DataFrame(rows)
df_out.to_csv(output_csv, index=False)
print(f"\nSummary CSV written to: {output_csv}")
