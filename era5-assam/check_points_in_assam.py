"""
check_points_in_assam.py
========================
Downloads the India state boundaries GeoJSON, extracts the Assam polygon,
and tests all 129 ERA5 population-weighted grid points for containment.

Outputs:
  assam_grid_points.csv         — points confirmed inside Assam
  outside_assam_grid_points.csv — points that fall outside Assam polygon

Usage:
  python check_points_in_assam.py
"""

import os
import sys
import csv
import json
import urllib.request
from pathlib import Path

# PATHS
HERE = Path(__file__).resolve().parent

GRID_POINTS_CSV = HERE / "data" / "processed" / "population_grid_points.csv"
GEOJSON_FILE    = HERE / "data" / "raw" / "boundary" / "india_states.geojson"

OUT_INSIDE  = HERE / "assam_grid_points.csv"
OUT_OUTSIDE = HERE / "outside_assam_grid_points.csv"

GEOJSON_FILE.parent.mkdir(parents=True, exist_ok=True)

# GeoJSON sources (tried in order)
# Primary: individual Assam state GeoJSON from udit-001/india-maps-data
GEOJSON_URLS = [
    # Assam-only GeoJSON (fastest)
    "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/states/assam.geojson",
    # All-India GeoJSON fallback
    "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson",
]


def download_geojson():
    if GEOJSON_FILE.exists() and GEOJSON_FILE.stat().st_size > 10_000:
        print(f"[SKIP] GeoJSON already present: {GEOJSON_FILE}")
        return
    for url in GEOJSON_URLS:
        print(f"Downloading boundary GeoJSON ...")
        print(f"  URL: {url}")
        try:
            urllib.request.urlretrieve(url, GEOJSON_FILE)
            print(f"  Saved to {GEOJSON_FILE}  ({GEOJSON_FILE.stat().st_size/1e3:.0f} KB)")
            return
        except Exception as exc:
            print(f"  [WARN] Failed ({exc}) — trying next URL ...")
            if GEOJSON_FILE.exists():
                GEOJSON_FILE.unlink()
    sys.exit("[ERROR] All GeoJSON download URLs failed. Check your internet connection.")


def load_assam_polygon_json():
    with open(GEOJSON_FILE, encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    if not features:
        sys.exit("[ERROR] GeoJSON has no features.")

    # All known property keys that could hold the state name
    STATE_NAME_KEYS = (
        "NAME_1", "ST_NM", "st_nm", "state", "name", "State",
        "NAME", "statename", "STATENAME", "st_name",
    )

    # Try to filter features whose state-name field contains 'assam'
    assam_features = [
        f for f in features
        if any(
            "assam" in str(f.get("properties", {}).get(k, "")).lower()
            for k in STATE_NAME_KEYS
        )
    ]

    # If every feature matches (single-state file like the district GeoJSON)
    if not assam_features:
        sample_props = features[0].get("properties", {})
        sys.exit(
            f"[ERROR] Could not find 'Assam' in the GeoJSON.\n"
            f"Total features: {len(features)}\n"
            f"Sample properties of first feature: {sample_props}\n"
            f"Checked keys: {STATE_NAME_KEYS}"
        )

    print(f"  Found {len(assam_features)} Assam feature(s) "
          f"(district polygons will be unioned into one state boundary).")
    return assam_features


def run_with_geopandas(assam_features, points):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union

    assam_geom = unary_union([shape(f["geometry"]) for f in assam_features])
    df = pd.DataFrame(points)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(r["lon"], r["lat"]) for r in points],
        crs="EPSG:4326"
    )
    gdf["Inside_Assam"] = gdf.geometry.within(assam_geom)
    return gdf


def run_with_shapely(assam_features, points):
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union

    assam_geom = unary_union([shape(f["geometry"]) for f in assam_features])
    results = []
    for r in points:
        pt = Point(r["lon"], r["lat"])
        results.append({**r, "Inside_Assam": assam_geom.contains(pt)})
    return results


def run_pip(assam_features, points):
    try:
        import geopandas
        print("  Using: geopandas + shapely")
        return run_with_geopandas(assam_features, points), "geopandas"
    except ImportError:
        pass
    try:
        from shapely.geometry import Point
        print("  Using: shapely only")
        return run_with_shapely(assam_features, points), "shapely"
    except ImportError:
        sys.exit("[ERROR] Install geopandas or shapely:\n  pip install geopandas shapely")


def load_grid_points():
    if not GRID_POINTS_CSV.exists():
        sys.exit(f"[ERROR] Grid points CSV not found: {GRID_POINTS_CSV}")
    rows = []
    with open(GRID_POINTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "point_id":   r["point_id"],
                "lat":        float(r["lat"]),
                "lon":        float(r["lon"]),
                "population": float(r["population"]),
                "weight":     float(r["weight"]),
            })
    return rows


def save_csv(rows, filepath):
    if not rows:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("point_id,lat,lon,population,weight,Inside_Assam\n")
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    print("\n" + "=" * 60)
    print("  Assam Grid-Point Boundary Verification")
    print("=" * 60)

    download_geojson()

    points = load_grid_points()
    print(f"\nLoaded {len(points)} grid points.")

    print("\nExtracting Assam polygon ...")
    assam_features = load_assam_polygon_json()

    print("\nRunning point-in-polygon test ...")
    result, engine = run_pip(assam_features, points)

    if engine == "geopandas":
        df = result.drop(columns=["geometry"])
        all_rows     = df.to_dict(orient="records")
        inside_rows  = df[df["Inside_Assam"]].to_dict(orient="records")
        outside_rows = df[~df["Inside_Assam"]].to_dict(orient="records")
    else:
        all_rows     = result
        inside_rows  = [r for r in result if r["Inside_Assam"]]
        outside_rows = [r for r in result if not r["Inside_Assam"]]

    print("\n" + "=" * 60)
    print(f"  Total Grid Points  : {len(all_rows)}")
    print(f"  Inside  Assam      : {len(inside_rows)}")
    print(f"  Outside Assam      : {len(outside_rows)}")
    print("=" * 60)

    if outside_rows:
        print(f"\n  Points OUTSIDE Assam ({len(outside_rows)}):")
        print(f"  {'ID':<12}  {'Lat':>8}  {'Lon':>8}")
        print("  " + "-" * 34)
        for r in outside_rows:
            pid = r.get("point_id", "?")
            print(f"  {pid:<12}  {r['lat']:>8.3f}  {r['lon']:>8.3f}")
    else:
        print("\n  All 129 points are confirmed INSIDE Assam!")

    save_csv(inside_rows,  OUT_INSIDE)
    save_csv(outside_rows, OUT_OUTSIDE)

    print(f"\n  Saved: {OUT_INSIDE.name}  ({len(inside_rows)} rows)")
    print(f"  Saved: {OUT_OUTSIDE.name}  ({len(outside_rows)} rows)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
