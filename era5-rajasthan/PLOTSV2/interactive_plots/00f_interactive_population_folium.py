"""Interactive population-grid Folium map.

Run with: streamlit run 00f_interactive_population_folium.py

Same shape as 03f_interactive_raw_folium.py. The population grid is one row per
sampling point with no date column, so it has no year/month/date selectors —
that is forced by the data, not a style choice.
"""
import sys
from pathlib import Path

import pandas as pd
import folium
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import POPULATION_GRID_FILE, PLOTS_DIR


@st.cache_data
def load_data(input_file):
    df = pd.read_csv(input_file)
    return df


def main(input_file=POPULATION_GRID_FILE, title="Population grid"):
    st.set_page_config(page_title=f"{title} Folium Explorer", layout="wide")
    df = load_data(str(input_file))
    excluded = {"point_id", "lat", "lon"}
    parameters = [c for c in df.select_dtypes(include="number").columns if c not in excluded]
    st.title(f"Rajasthan {title}: Folium explorer")
    parameter = st.selectbox("Parameter", parameters, index=parameters.index("population") if "population" in parameters else 0)
    fmap = folium.Map(location=[df.lat.mean(), df.lon.mean()], zoom_start=7, tiles="OpenStreetMap")
    for _, row in df.iterrows():
        lines = "<br>".join(f"{c}: {row[c] if pd.notna(row[c]) else 'missing'}" for c in parameters)
        folium.CircleMarker([row.lat, row.lon], radius=6, color="#2563eb", fill=True, fill_opacity=.8,
                            popup=folium.Popup(f"<b>{row.point_id}</b><br><b>{parameter}</b><br>{lines}", max_width=300)).add_to(fmap)
    # Tamil Nadu uses st.components.v1.html here; Streamlit now warns that it
    # "will be removed after 2026-06-01" and to use st.iframe, which takes the
    # same HTML string.
    st.iframe(fmap.get_root().render(), height=700)


if __name__ == "__main__":
    main()
