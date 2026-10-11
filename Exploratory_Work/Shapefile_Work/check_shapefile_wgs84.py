import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

SRC_SHP = "/data/users/bob.potts/sowf_data/UK-Countries/CTRY_DEC_2021_UK_BUC.shp"
OUT_SHP = SRC_SHP.replace(".shp", "_wgs84.shp")
NAME_COLUMN = "CTRY21NM"
PREFIXED = ["England", "Scotland", "Wales", "Northern Ireland"]

gdf = gpd.read_file(SRC_SHP)
print(f"Source CRS: {gdf.crs}")
print(f"Source features: {len(gdf)}")
print(f"Source names: {gdf[NAME_COLUMN].tolist()}")
print(f"Source bounds: {gdf.total_bounds}")

ax = gdf.plot(column=NAME_COLUMN, legend=True, edgecolor="black", linewidth=0.5, figsize=(6, 8))
ax.set_title("Original file (EPSG:27700, metres)")
plt.show()

expected_names = [f"UK-{n}" if n in PREFIXED else n for n in gdf[NAME_COLUMN]]
gdf_wgs84 = gdf.to_crs(epsg=4326)
gdf_wgs84[NAME_COLUMN] = expected_names
gdf_wgs84.to_file(OUT_SHP)

gdf_check = gpd.read_file(OUT_SHP)
minx, miny, maxx, maxy = gdf_check.total_bounds
print(f"Output CRS: {gdf_check.crs}")
print(f"Output features: {len(gdf_check)}")
print(f"Output names: {gdf_check[NAME_COLUMN].tolist()}")
print(f"Output bounds: {gdf_check.total_bounds}")

assert gdf_check.crs.to_epsg() == 4326, "Output CRS is not EPSG:4326."
assert len(gdf_check) > 0, "Output file has no features."
assert len(gdf_check) == len(gdf), "Feature count changed."
assert gdf_check[NAME_COLUMN].tolist() == expected_names, "Names are wrong."
assert gdf_check.is_valid.all(), "Output has invalid geometries."
assert not gdf_check.is_empty.any(), "Output has empty geometries."
assert -9 < minx and maxx < 2 and 49 < miny and maxy < 61, "Bounds are outside the UK."
print("All checks passed.")

fig, ax = plt.subplots(figsize=(6, 8), subplot_kw={"projection": ccrs.PlateCarree()})
gdf_check.plot(
    ax=ax, column=NAME_COLUMN, legend=True, alpha=0.5, edgecolor="black", linewidth=0.5,
    transform=ccrs.PlateCarree(),
)
ax.coastlines(resolution="10m", color="red", linewidth=0.8)
ax.set_extent([minx - 0.5, maxx + 0.5, miny - 0.5, maxy + 0.5], crs=ccrs.PlateCarree())
ax.gridlines(draw_labels=True)
ax.set_title("Re-read file (EPSG:4326, degrees) with Natural Earth coastline in red")
plt.show()
