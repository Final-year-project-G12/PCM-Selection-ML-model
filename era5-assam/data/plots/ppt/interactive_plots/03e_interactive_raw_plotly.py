"""Interactive raw-data Plotly explorer for Assam.

Run with: streamlit run 03e_interactive_raw_plotly.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PREPROCESSED_DIR, PLOTS_DIR

COMBINED_POINTS_FILE = PREPROCESSED_DIR / "assam_cleaned_physical.csv"


@st.cache_data
def load_data(input_file):
    df = pd.read_csv(input_file, parse_dates=["date"], nrows=500000)
    return df


def main(input_file=COMBINED_POINTS_FILE, title="Raw"):
    st.set_page_config(page_title=f"{title} Plotly Explorer - Assam", layout="wide")
    df = load_data(str(input_file))
    excluded = {"point_id", "lat", "lon", "date", "event", "time_utc", "season", "grid_lat", "grid_lon", "population", "weight", "year", "month", "DOY", "season_code"}
    parameters = [c for c in df.select_dtypes(include="number").columns if c not in excluded]
    st.title(f"Assam {title} data: Plotly explorer")
    year = st.selectbox("Year", sorted(df.year.unique()))
    month = st.selectbox("Month", sorted(df.loc[df.year == year, "month"].unique()))
    dates = sorted(df.loc[(df.year == year) & (df.month == month), "date"].dt.strftime("%Y-%m-%d").unique())
    date_text = st.selectbox("Date", dates)
    parameter = st.selectbox("Parameter", parameters, index=parameters.index("era5_GHI") if "era5_GHI" in parameters else 0)
    selected = df[df.date.dt.strftime("%Y-%m-%d") == date_text]
    fig = px.scatter(selected, x="lon", y="lat", color=parameter, hover_name="point_id", hover_data=["event", parameter], color_continuous_scale="Viridis", title=f"{parameter} on {date_text}")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
