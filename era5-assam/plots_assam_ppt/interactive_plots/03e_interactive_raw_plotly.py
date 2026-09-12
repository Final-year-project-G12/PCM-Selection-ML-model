import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure repository root and era5-assam folder are in sys.path
_current_dir = Path(__file__).resolve().parent
for parent in (_current_dir.parent, _current_dir.parent.parent):
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

try:
    from config import PREPROCESSED_DIR
    DEFAULT_INPUT_FILE = PREPROCESSED_DIR / "assam_cleaned_physical.csv"
except Exception:
    DEFAULT_INPUT_FILE = Path(__file__).resolve().parents[2] / "data" / "preprocessed" / "assam_cleaned_physical.csv"


@st.cache_data(show_spinner=True)
def load_data(input_file_path: str):
    """Load cleaned physical data with memory-efficient types."""
    p = Path(input_file_path)
    if not p.exists():
        return None
    # Load first 500,000 rows for interactive exploration
    df = pd.read_csv(p, parse_dates=["date"], nrows=500000)
    return df


def main(input_file=DEFAULT_INPUT_FILE, title="Raw"):
    st.set_page_config(page_title=f"{title} Plotly Explorer - Assam", layout="wide")
    
    file_path = Path(input_file)
    if not file_path.exists():
        st.error(f"Data file not found at: `{file_path}`\nPlease verify that preprocessing has been executed.")
        return

    st.title(f"Assam {title} Climate Data Explorer")
    st.caption("Interactive spatial Plotly visualizer for ERA5 atmospheric and solar parameters.")

    with st.spinner("Loading climate dataset..."):
        df = load_data(str(file_path))

    if df is None or df.empty:
        st.error("Dataset is empty or could not be loaded.")
        return

    # Identify numerical parameters
    excluded = {
        "point_id", "lat", "lon", "date", "event", "time_utc", "season",
        "grid_lat", "grid_lon", "population", "weight", "year", "month",
        "DOY", "season_code", "is_daytime"
    }
    parameters = [c for c in df.select_dtypes(include="number").columns if c not in excluded]
    
    if not parameters:
        st.error("No numerical climate parameters found in dataset.")
        return

    # Controls Layout
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        years = sorted(df["year"].dropna().unique())
        selected_year = st.selectbox("Year", years, index=0 if years else None)

    with c2:
        available_months = sorted(df.loc[df["year"] == selected_year, "month"].dropna().unique())
        selected_month = st.selectbox("Month", available_months, index=0 if available_months else None)

    with c3:
        matching_dates = sorted(
            df.loc[(df["year"] == selected_year) & (df["month"] == selected_month), "date"]
            .dt.strftime("%Y-%m-%d")
            .dropna()
            .unique()
        )
        if not matching_dates:
            st.warning("No dates found for selected year and month.")
            return
        selected_date_str = st.selectbox("Date", matching_dates)

    with c4:
        # Event filter to avoid 3 overlapping points (sunrise, noon, sunset) at identical coordinates
        available_events = sorted(df["event"].dropna().unique().tolist())
        selected_event = st.selectbox("Diurnal Event", ["All Events"] + available_events, index=0)

    # Filter data
    date_mask = df["date"].dt.strftime("%Y-%m-%d") == selected_date_str
    if selected_event != "All Events":
        date_mask = date_mask & (df["event"] == selected_event)
    
    selected_df = df[date_mask].copy()

    # Parameter selection
    default_param_idx = parameters.index("era5_GHI") if "era5_GHI" in parameters else 0
    param = st.selectbox("Parameter to Visualize", parameters, index=default_param_idx)

    # Spatial plot
    if not selected_df.empty:
        hover_cols = ["point_id", "event", param, "lat", "lon"]
        hover_cols = [c for c in hover_cols if c in selected_df.columns]
        
        fig = px.scatter_mapbox(
            selected_df,
            lat="lat",
            lon="lon",
            color=param,
            hover_name="point_id",
            hover_data={c: True for c in hover_cols},
            color_continuous_scale="Viridis",
            zoom=6.2,
            center={"lat": float(selected_df["lat"].mean()), "lon": float(selected_df["lon"].mean())},
            mapbox_style="carto-positron",
            title=f"{param} on {selected_date_str} ({selected_event})"
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=600)
        
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.plotly_chart(fig)
            
        st.caption(f"Displaying {len(selected_df)} observations across Assam.")
    else:
        st.info(f"No records matching {selected_date_str} and event '{selected_event}'.")


if __name__ == "__main__":
    # If executed directly with standard python (e.g. VS Code 'Run Python File'),
    # automatically launch with Streamlit instead of crashing with missing context.
    import streamlit.runtime
    if not streamlit.runtime.exists():
        print("Launching via Streamlit...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        main()

