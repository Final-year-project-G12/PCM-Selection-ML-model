# Rajasthan interactive explorers

1:1 ports of the Tamil Nadu explorers in
`era5-tamilnadu/plots_tamilnadu_ppt/interactive_plots/` — same filenames, same
code structure, same widgets, same `main(input_file, title)` + wrapper pattern.
Only the paths and the state name differ.

| File | Library | Data |
| :--- | :--- | :--- |
| `00e_interactive_population_plotly.py` | Plotly | `POPULATION_GRID_FILE` — `population_grid_points.csv` |
| `00f_interactive_population_folium.py` | Folium | `POPULATION_GRID_FILE` — `population_grid_points.csv` |
| `03e_interactive_raw_plotly.py` | Plotly | `COMBINED_POINTS_FILE` — `climate_rajasthan_points.csv` |
| `03f_interactive_raw_folium.py` | Folium | `COMBINED_POINTS_FILE` — `climate_rajasthan_points.csv` |
| `04e_interactive_preprocessed_plotly.py` | Plotly | `PREPROCESSED_DIR/rajasthan_cleaned_physical.csv` |
| `04f_interactive_preprocessed_folium.py` | Folium | `PREPROCESSED_DIR/rajasthan_cleaned_physical.csv` |

Naming follows the Tamil Nadu convention `<step><letter>_interactive_<dataset>_<library>.py`,
with `e` = Plotly and `f` = Folium. The `03e/03f/04e/04f` pairs match Tamil Nadu
file-for-file; `00e/00f` are the extra population-grid pair (Tamil Nadu has no
equivalent), numbered `00` after `00a_build_population_grid.py`, the script that
produces that file.

## Running

```
streamlit run 03e_interactive_raw_plotly.py
streamlit run 03f_interactive_raw_folium.py
streamlit run 04e_interactive_preprocessed_plotly.py
streamlit run 04f_interactive_preprocessed_folium.py
streamlit run 00e_interactive_population_plotly.py
streamlit run 00f_interactive_population_folium.py
```

Add `--server.port 8502` to run a second one alongside the first.

## Three deliberate differences from the Tamil Nadu files

Everything else is a straight copy.

1. **`sys.path` shim before `from config import ...`.** The Tamil Nadu files sit
   two levels below their pipeline root and import `config` directly, which only
   resolves when the app is launched from the pipeline root. These add two lines
   putting `era5-rajasthan/` on the path, so they run from this folder.
2. **`st.iframe` instead of `st.components.v1.html`** in the Folium apps.
   Streamlit now warns that `st.components.v1.html` "will be removed after
   2026-06-01"; `st.iframe` takes the same HTML string.
3. **`width="stretch"` instead of `use_container_width=True`** in
   `st.plotly_chart`. `use_container_width` is deprecated.

The population pair also has no year/month/date selectors — that file is one row
per sampling point with no date column, so those selectors cannot exist.

## Load times

These follow Tamil Nadu in reading the whole CSV with `pd.read_csv` inside
`@st.cache_data`. That is cheap for Tamil Nadu's files but not for Rajasthan's,
which are several times larger. Measured on this machine, first load per session:

| App | File size | First load |
| :--- | ---: | ---: |
| `00e` / `00f` | 25 KB | ~1 s |
| `03e` / `03f` | 1.42 GB | ~27 s |
| `04e` / `04f` | 3.94 GB | ~80 s |

Subsequent interactions are instant — `@st.cache_data` holds the frame for the
rest of the session. The cost is paid again on every fresh `streamlit run`.

The `04e`/`04f` pair peaks at roughly 3 GB of RAM. It completed here with about
420 MB physically free (Windows paged the rest), but on a loaded machine expect
it to be slow. If that becomes a problem, a dtype-shrink plus Parquet cache cuts
the clean-points file from 34 s / 1532 MB to 1.2 s / 445 MB — ask and I can add
it back, at the cost of no longer being a literal copy of the Tamil Nadu code.

## Verification

All six pass a headless `streamlit.testing.v1.AppTest` render — no exceptions,
no warnings, all expected selectors present.
