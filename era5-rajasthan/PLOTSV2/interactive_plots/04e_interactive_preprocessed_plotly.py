"""Interactive preprocessed-data Plotly explorer.

Run with: streamlit run 04e_interactive_preprocessed_plotly.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import PREPROCESSED_DIR, PLOTS_DIR
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("raw_plotly", __file__.replace("04e_interactive_preprocessed_plotly.py", "03e_interactive_raw_plotly.py"))
raw_plotly = module_from_spec(spec)
spec.loader.exec_module(raw_plotly)

if __name__ == "__main__":
    raw_plotly.main(PREPROCESSED_DIR / "rajasthan_cleaned_physical.csv", "Preprocessed")
