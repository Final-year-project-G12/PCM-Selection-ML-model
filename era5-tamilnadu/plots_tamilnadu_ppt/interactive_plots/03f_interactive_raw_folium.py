"""Interactive raw-data Folium map.

Run with: streamlit run 03f_interactive_raw_folium.py
"""
import pandas as pd
import folium
import streamlit as st
import streamlit.components.v1 as components

from config import COMBINED_POINTS_FILE, PLOTS_DIR


@st.cache_data
def load_data(input_file):
    df = pd.read_csv(input_file, parse_dates=["date"])
    return df


def main(input_file=COMBINED_POINTS_FILE, title="Raw"):
    st.set_page_config(page_title=f"{title} Folium Explorer", layout="wide")
    df = load_data(str(input_file))
    excluded = {"point_id", "lat", "lon", "date", "date_text", "event", "time_utc", "season",
                "grid_lat", "grid_lon", "population", "weight", "year", "month", "DOY", "season_code"}
    parameters = [c for c in df.select_dtypes(include="number").columns if c not in excluded]
    st.title(f"Tamil Nadu {title} data: Folium explorer")
    year = st.selectbox("Year", sorted(df.year.unique()))
    month = st.selectbox("Month", sorted(df.loc[df.year == year, "month"].unique()))
    dates = sorted(df.loc[(df.year == year) & (df.month == month), "date"].dt.strftime("%Y-%m-%d").unique())
    date_text = st.selectbox("Date", dates)
    parameter = st.selectbox("Parameter", parameters, index=parameters.index("era5_GHI") if "era5_GHI" in parameters else 0)
    selected = df[df.date.dt.strftime("%Y-%m-%d") == date_text]
    fmap = folium.Map(location=[df.lat.mean(), df.lon.mean()], zoom_start=7, tiles="OpenStreetMap")
    for point_id, group in selected.groupby("point_id"):
        first = group.iloc[0]
        lines = "<br>".join(f"{row.event}: {row[parameter] if pd.notna(row[parameter]) else 'missing'}" for _, row in group.iterrows())
        folium.CircleMarker([first.lat, first.lon], radius=6, color="#2563eb", fill=True, fill_opacity=.8,
                            popup=folium.Popup(f"<b>{point_id}</b><br><b>{parameter}</b> on {date_text}<br>{lines}", max_width=300)).add_to(fmap)
    components.html(fmap.get_root().render(), height=700, scrolling=False)


if __name__ == "__main__":
    main()