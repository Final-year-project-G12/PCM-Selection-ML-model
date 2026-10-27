"""Interactive preprocessed-data Folium map for Assam.

Run with: streamlit run 04f_interactive_preprocessed_folium.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PREPROCESSED_DIR
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("raw_folium", __file__.replace("04f_interactive_preprocessed_folium.py", "03f_interactive_raw_folium.py"))
raw_folium = module_from_spec(spec)
spec.loader.exec_module(raw_folium)

if __name__ == "__main__":
    raw_folium.main(PREPROCESSED_DIR / "assam_cleaned_physical.csv", "Preprocessed")
