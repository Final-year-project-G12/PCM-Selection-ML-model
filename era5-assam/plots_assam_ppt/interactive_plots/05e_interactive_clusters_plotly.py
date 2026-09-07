"""Interactive K=3 Cluster & PCM Recommendation Explorer for Assam.

Run with: streamlit run 05e_interactive_clusters_plotly.py
"""
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE_DIR = Path(__file__).resolve().parents[2]
POP_CSV = BASE_DIR / "data" / "processed" / "population_grid_points.csv"
CLUSTER_ASSIGN = BASE_DIR / "data" / "processed" / "clustering" / "cluster_assignments_assam.csv"
CLUSTER_PROFILES = BASE_DIR / "data" / "processed" / "clustering" / "cluster_profiles_assam.csv"
TOPK_CSV = BASE_DIR / "data" / "processed" / "pcm" / "mcdm_topk_assam.csv"


@st.cache_data
def load_cluster_data():
    pop_df = pd.read_csv(POP_CSV)
    clus_df = pd.read_csv(CLUSTER_ASSIGN)
    merged = clus_df.merge(pop_df[["point_id", "lat", "lon", "population", "weight"]], on="point_id")
    
    cluster_names = {
        0: "Cluster 0 (Moderate Valley)",
        1: "Cluster 1 (Humid Subtropical)",
        2: "Cluster 2 (Highland / Cool)"
    }
    merged["cluster_name"] = merged["cluster"].map(cluster_names)
    
    profiles = pd.read_csv(CLUSTER_PROFILES) if CLUSTER_PROFILES.exists() else None
    topk = pd.read_csv(TOPK_CSV) if TOPK_CSV.exists() else None
    return merged, profiles, topk


def main():
    st.set_page_config(page_title="Assam K=3 Climate Regimes & PCM Explorer", layout="wide")
    st.title("Assam K=3 Climate Regimes & PCM Selection Explorer")
    st.markdown(
        "Interactive exploration of the **K=3 Gaussian Mixture Model (GMM) Climate Regimes** "
        "and corresponding multi-criteria phase change material (PCM) recommendations."
    )

    merged, profiles, topk = load_cluster_data()

    # Sidebar controls
    st.sidebar.header("Filter & Settings")
    selected_clusters = st.sidebar.multiselect(
        "Select Clusters to Display",
        options=[0, 1, 2],
        default=[0, 1, 2],
        format_func=lambda c: f"Cluster {c}"
    )

    prob_threshold = st.sidebar.slider("Minimum Membership Probability", 0.0, 1.0, 0.5, 0.05)

    filtered_df = merged[
        merged["cluster"].isin(selected_clusters) &
        (merged["max_membership_prob"] >= prob_threshold)
    ]

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Assam Climate Regimes (K=3 Spatial Distribution)")
        color_map = {
            "Cluster 0 (Moderate Valley)": "#1f77b4",
            "Cluster 1 (Humid Subtropical)": "#ff7f0e",
            "Cluster 2 (Highland / Cool)": "#2ca02c"
        }

        fig = px.scatter_mapbox(
            filtered_df,
            lat="lat",
            lon="lon",
            color="cluster_name",
            color_discrete_map=color_map,
            size="population",
            hover_name="point_id",
            hover_data={
                "cluster": True,
                "max_membership_prob": ":.3f",
                "lat": ":.3f",
                "lon": ":.3f",
                "population": ":,",
                "cluster_name": False
            },
            zoom=6.3,
            center={"lat": 26.2, "lon": 92.8},
            mapbox_style="carto-positron",
            title=f"129 Grid Points Categorized into K=3 Regimes (Showing {len(filtered_df)} points)"
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=550)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cluster Distribution")
        dist = merged["cluster_name"].value_counts().reset_index()
        dist.columns = ["Regime", "Point Count"]
        fig_pie = px.pie(
            dist,
            names="Regime",
            values="Point Count",
            color="Regime",
            color_discrete_map=color_map,
            hole=0.4
        )
        fig_pie.update_layout(margin={"r": 0, "t": 20, "l": 0, "b": 0}, height=280)
        st.plotly_chart(fig_pie, use_container_width=True)

        if profiles is not None:
            st.subheader("Regime Profiles (Summary)")
            display_cols = [c for c in ["cluster_id", "Ta_mean", "GHI_mean", "RH_mean", "Tm_target"] if c in profiles.columns]
            st.dataframe(profiles[display_cols].rename(columns={"cluster_id": "Cluster"}), use_container_width=True)

    st.markdown("---")
    st.header("Top-3 PCM Recommendations per Cluster (MCDM Consensus)")

    if topk is not None:
        tabs = st.tabs(["Cluster 0 (Moderate Valley)", "Cluster 1 (Humid Subtropical)", "Cluster 2 (Highland / Cool)"])
        for cid, tab in enumerate(tabs):
            with tab:
                c_topk = topk[topk["cluster_id"] == cid].sort_values("consensus_rank")
                if not c_topk.empty:
                    st.write(f"**Target Melting Temperature ($T_{{m,\\text{{target}}}}$):** {c_topk['Tm_target'].iloc[0]:.1f} °C" if "Tm_target" in c_topk.columns else "")
                    cols = [c for c in ["consensus_rank", "name", "Tm", "latent_heat", "thermal_conductivity", "density", "topsis_rank", "vikor_rank"] if c in c_topk.columns]
                    st.dataframe(c_topk[cols].reset_index(drop=True), use_container_width=True)
                else:
                    st.info(f"No recommendations found for Cluster {cid}.")


if __name__ == "__main__":
    main()
