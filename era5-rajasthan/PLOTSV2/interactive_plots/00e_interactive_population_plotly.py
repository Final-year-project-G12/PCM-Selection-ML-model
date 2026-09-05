"""Interactive population-grid Plotly explorer.

Run with: streamlit run 00e_interactive_population_plotly.py

Same shape as 03e_interactive_raw_plotly.py. The population grid is one row per
sampling point with no date column, so it has no year/month/date selectors —
that is forced by the data, not a style choice.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import POPULATION_GRID_FILE, PLOTS_DIR


@st.cache_data
def load_data(input_file):
    df = pd.read_csv(input_file)
    return df


def main(input_file=POPULATION_GRID_FILE, title="Population grid"):
    st.set_page_config(page_title=f"{title} Plotly Explorer", layout="wide")
    df = load_data(str(input_file))
    excluded = {"point_id", "lat", "lon"}
    parameters = [c for c in df.select_dtypes(include="number").columns if c not in excluded]
    st.title(f"Rajasthan {title}: Plotly explorer")
    parameter = st.selectbox("Parameter", parameters, index=parameters.index("population") if "population" in parameters else 0)
    fig = px.scatter(df, x="lon", y="lat", color=parameter, size="population", size_max=24, hover_name="point_id", hover_data=parameters, color_continuous_scale="Viridis", title=f"{parameter} across the sampling grid")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
