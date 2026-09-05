"""
Final MCDM cluster outputs for presentation and inspection.

Reads the complete MCDM score table and cluster assignments, then writes:
  data/processed/pcm/mcdm_final_results_complete.csv
  data/plots/mcdm/mcdm_clusters_plotly.html
  data/plots/mcdm/mcdm_clusters_folium.html

The CSV retains every column from mcdm_full_scores_by_cluster.csv and adds
the cluster representative coordinates and population summary needed by the
maps. Map popups/hover text expose the complete row rather than a reduced
selection of score columns.

Run after 08_mcdm_ranking.py:
  python 12_mcdm_interactive_plots.py
"""

from html import escape

import folium
import pandas as pd
import plotly.express as px

from config import PLOTS_DIR, PROCESSED_DIR

PCM_DIR = PROCESSED_DIR / "pcm"
CLUSTER_DIR = PROCESSED_DIR / "clustering"
MCDM_FILE = PCM_DIR / "mcdm_full_scores_by_cluster.csv"
ASSIGN_FILE = CLUSTER_DIR / "cluster_assignments_tamilnadu.csv"
OUT_DIR = PLOTS_DIR / "mcdm"
OUT_CSV = PCM_DIR / "mcdm_final_results_complete.csv"
OUT_PLOTLY = OUT_DIR / "mcdm_clusters_plotly.html"
OUT_FOLIUM = OUT_DIR / "mcdm_clusters_folium.html"


def format_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def full_row_html(row):
    fields = "".join(
        f"<tr><th style='text-align:left;padding-right:8px'>{escape(str(column))}</th>"
        f"<td>{escape(format_value(value))}</td></tr>"
        for column, value in row.items()
    )
    return f"<table style='font-size:11px'>{fields}</table>"


def main():
    for path in (MCDM_FILE, ASSIGN_FILE):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run the earlier pipeline step first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mcdm = pd.read_csv(MCDM_FILE)
    assignments = pd.read_csv(ASSIGN_FILE)

    cluster_locations = (assignments.groupby("cluster_id", as_index=False)
                         .agg(cluster_lat=("lat", "mean"),
                              cluster_lon=("lon", "mean"),
                              cluster_population=("population", "sum"),
                              cluster_point_count=("point_id", "nunique")))
    final = mcdm.merge(cluster_locations, on="cluster_id", how="left", validate="many_to_one")
    final = final.sort_values(["cluster_id", "consensus_rank", "name"]).reset_index(drop=True)
    final.to_csv(OUT_CSV, index=False)

    # One marker per sampled population point, annotated with that cluster's
    # complete ranked result table in the map detail view.
    top1 = (final[final["consensus_rank"] == 1]
            .sort_values(["cluster_id", "name"])
            .drop_duplicates("cluster_id"))
    top1_fields = top1[["cluster_id", "name", "family", "consensus_rank"]].rename(
        columns={"name": "recommended_pcm", "family": "recommended_family",
                 "consensus_rank": "recommended_rank"})
    map_points = assignments.merge(top1_fields, on="cluster_id", how="left")
    map_points.to_csv(OUT_DIR / "mcdm_cluster_map_points.csv", index=False)

    full_by_cluster = {
        int(cluster_id): group.sort_values("consensus_rank")
        for cluster_id, group in final.groupby("cluster_id")
    }

    palette = ["#d1495b", "#00798c", "#edae49", "#30638e", "#003d5b",
               "#6a994e", "#9b5de5", "#f15bb5"]
    folium_map = folium.Map(
        location=[assignments["lat"].mean(), assignments["lon"].mean()],
        zoom_start=7,
        tiles="OpenStreetMap",
    )
    for row in map_points.itertuples(index=False):
        cluster_rows = full_by_cluster.get(int(row.cluster_id), pd.DataFrame())
        popup = (f"<b>{escape(str(row.point_id))}</b><br>"
                 f"Cluster: {int(row.cluster_id)}<br>"
                 f"Recommended PCM: {escape(str(row.recommended_pcm))}<br>"
                 f"Population: {row.population:,.0f}<hr>"
                 f"{full_row_html(cluster_rows.iloc[0]) if len(cluster_rows) else ''}")
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color=palette[int(row.cluster_id) % len(palette)],
            fill=True,
            fill_color=palette[int(row.cluster_id) % len(palette)],
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=520),
            tooltip=f"{row.point_id} | Cluster {int(row.cluster_id)} | {row.recommended_pcm}",
        ).add_to(folium_map)
    folium.LayerControl().add_to(folium_map)
    folium_map.save(OUT_FOLIUM)

    hover_columns = ["point_id", "cluster_id", "recommended_pcm", "recommended_family",
                     "population", "max_membership_prob"]
    hover_labels = {
        "point_id": "Point", "cluster_id": "Cluster", "recommended_pcm": "Recommended PCM",
        "recommended_family": "Family", "population": "Population",
        "max_membership_prob": "Membership confidence",
    }
    plot = px.scatter_mapbox(
        map_points,
        lat="lat",
        lon="lon",
        color="cluster_id",
        category_orders={"cluster_id": sorted(map_points["cluster_id"].unique())},
        hover_name="point_id",
        hover_data={column: True for column in hover_columns if column != "point_id"},
        labels=hover_labels,
        zoom=6,
        center={"lat": map_points["lat"].mean(), "lon": map_points["lon"].mean()},
        title="Tamil Nadu MCDM PCM Recommendations by Climate Cluster",
        color_discrete_sequence=palette,
    )
    plot.update_layout(mapbox_style="open-street-map", margin={"l": 0, "r": 0, "t": 45, "b": 0})
    plot.write_html(OUT_PLOTLY, include_plotlyjs=True)

    print(f"Saved complete CSV: {OUT_CSV}")
    print(f"Saved Plotly map: {OUT_PLOTLY}")
    print(f"Saved Folium map: {OUT_FOLIUM}")
    print(f"Saved map-point CSV: {OUT_DIR / 'mcdm_cluster_map_points.csv'}")


if __name__ == "__main__":
    main()