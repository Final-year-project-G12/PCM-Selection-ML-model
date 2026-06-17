"""
ERA5 COMBINE — TAMIL NADU  (Complete version)
==============================================
Reads  : data/raw/era5/grid/era5_TN_grid_{year}_{month}_{instant,accum}.nc
Writes :
  • data/processed/by_location/climate_{name}.csv   — one CSV per named location
  • data/processed/grid/era5_TN_grid_all.csv        — full 528-point grid CSV
  • data/processed/climate_tamilnadu_all.csv        — all named locations combined

Named locations: 260+ cities, towns, taluks across all 38 districts of Tamil Nadu
Grid points    : every ERA5 0.25° point inside TN bounding box (~528 points)

Requirements:
  pip install xarray netCDF4 pvlib pandas numpy scipy

HOW TO RUN:
  python 02_combine_tamilnadu.py

The script is safe to re-run — it will overwrite CSVs with fresh data.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import pvlib

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

INPUT_DIR        = "data/raw/era5/grid"
OUTPUT_NAMED     = "data/processed/by_location"
OUTPUT_GRID      = "data/processed/grid"
OUTPUT_COMBINED  = "data/processed"

for d in [OUTPUT_NAMED, OUTPUT_GRID, OUTPUT_COMBINED]:
    os.makedirs(d, exist_ok=True)

YEARS  = ["2024", "2025"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

RRTDHS_THRESHOLD = 0.5

SEASON_MAP = {
    12: ("Winter", 1),  1: ("Winter", 1),  2: ("Winter", 1),
     3: ("Summer", 2),  4: ("Summer", 2),  5: ("Summer", 2),
     6: ("Monsoon", 3), 7: ("Monsoon", 3), 8: ("Monsoon", 3),
     9: ("Retreat", 4), 10: ("Retreat", 4), 11: ("Retreat", 4),
}


# ═══════════════════════════════════════════════════════════
# ALL 260+ TAMIL NADU LOCATIONS
# ═══════════════════════════════════════════════════════════
# Format: "Name": {"lat":..., "lon":..., "alt":..., "dist":..., "cz":..., "T":...}
#   lat/lon  — decimal degrees
#   alt      — altitude in metres
#   dist     — district name
#   cz       — climate zone
#   T        — T_set (optimal panel temp for RRTDHS calculation)
# ═══════════════════════════════════════════════════════════

TN_LOCATIONS = {

    # ════════════════════════════════════════════════════
    # CHENNAI DISTRICT
    # ════════════════════════════════════════════════════
    "Chennai":           {"lat": 13.0827, "lon": 80.2707, "alt":   6, "dist": "Chennai",         "cz": "hot-humid-coastal",  "T": 45},
    "Tambaram":          {"lat": 12.9249, "lon": 80.1000, "alt":  25, "dist": "Chennai",         "cz": "hot-humid",          "T": 45},
    "Ambattur":          {"lat": 13.1143, "lon": 80.1548, "alt":  12, "dist": "Chennai",         "cz": "hot-humid",          "T": 45},
    "Pallavaram":        {"lat": 12.9675, "lon": 80.1491, "alt":  20, "dist": "Chennai",         "cz": "hot-humid",          "T": 45},
    "Avadi":             {"lat": 13.1067, "lon": 80.1053, "alt":  20, "dist": "Chennai",         "cz": "hot-humid",          "T": 45},
    "Ponneri":           {"lat": 13.3400, "lon": 80.1986, "alt":   5, "dist": "Tiruvallur",      "cz": "hot-humid-coastal",  "T": 45},
    "Ennore":            {"lat": 13.2167, "lon": 80.3167, "alt":   5, "dist": "Chennai",         "cz": "hot-humid-coastal",  "T": 45},

    # ════════════════════════════════════════════════════
    # TIRUVALLUR DISTRICT
    # ════════════════════════════════════════════════════
    "Tiruvallur":        {"lat": 13.1427, "lon": 79.9143, "alt":  20, "dist": "Tiruvallur",      "cz": "hot-humid",          "T": 45},
    "Gummidipoondi":     {"lat": 13.4058, "lon": 80.1155, "alt":   5, "dist": "Tiruvallur",      "cz": "hot-humid",          "T": 45},
    "Tiruttani":         {"lat": 13.1793, "lon": 79.6172, "alt": 100, "dist": "Tiruvallur",      "cz": "hot-humid",          "T": 45},
    "Poonamallee":       {"lat": 13.0468, "lon": 80.1166, "alt":  25, "dist": "Tiruvallur",      "cz": "hot-humid",          "T": 45},
    "Uthukottai":        {"lat": 13.4046, "lon": 79.9725, "alt":  10, "dist": "Tiruvallur",      "cz": "hot-humid",          "T": 45},

    # ════════════════════════════════════════════════════
    # CHENGALPATTU DISTRICT
    # ════════════════════════════════════════════════════
    "Chengalpattu":      {"lat": 12.6921, "lon": 79.9763, "alt":  45, "dist": "Chengalpattu",    "cz": "hot-humid",          "T": 45},
    "Mahabalipuram":     {"lat": 12.6269, "lon": 80.1927, "alt":   3, "dist": "Chengalpattu",    "cz": "hot-humid-coastal",  "T": 45},
    "Madurantakam":      {"lat": 12.4972, "lon": 79.8975, "alt":  15, "dist": "Chengalpattu",    "cz": "hot-humid",          "T": 45},
    "Uthiramerur":       {"lat": 12.5715, "lon": 79.7510, "alt":  50, "dist": "Chengalpattu",    "cz": "hot-humid",          "T": 45},
    "Cheyyur":           {"lat": 12.3247, "lon": 79.9310, "alt":   5, "dist": "Chengalpattu",    "cz": "hot-humid-coastal",  "T": 45},

    # ════════════════════════════════════════════════════
    # KANCHIPURAM DISTRICT
    # ════════════════════════════════════════════════════
    "Kanchipuram":       {"lat": 12.8185, "lon": 79.6947, "alt":  83, "dist": "Kanchipuram",     "cz": "hot-humid",          "T": 45},
    "Sriperumbudur":     {"lat": 12.9681, "lon": 79.9468, "alt":  50, "dist": "Kanchipuram",     "cz": "hot-humid",          "T": 45},
    "Walajabad":         {"lat": 12.7910, "lon": 79.8390, "alt":  60, "dist": "Kanchipuram",     "cz": "hot-humid",          "T": 45},
    "Uthiramerur_KCP":   {"lat": 12.5715, "lon": 79.7510, "alt":  50, "dist": "Kanchipuram",     "cz": "hot-humid",          "T": 45},

    # ════════════════════════════════════════════════════
    # VELLORE DISTRICT
    # ════════════════════════════════════════════════════
    "Vellore":           {"lat": 12.9165, "lon": 79.1325, "alt": 216, "dist": "Vellore",         "cz": "semi-arid",          "T": 50},
    "Arcot":             {"lat": 12.9038, "lon": 79.3165, "alt": 130, "dist": "Vellore",         "cz": "semi-arid",          "T": 50},
    "Gudiyatham":        {"lat": 12.9523, "lon": 78.8757, "alt": 220, "dist": "Vellore",         "cz": "semi-arid",          "T": 50},
    "Katpadi":           {"lat": 12.9714, "lon": 79.1437, "alt": 210, "dist": "Vellore",         "cz": "semi-arid",          "T": 50},
    "Arakkonam":         {"lat": 13.0765, "lon": 79.6714, "alt": 130, "dist": "Ranipet",         "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # RANIPET DISTRICT
    # ════════════════════════════════════════════════════
    "Ranipet":           {"lat": 12.9246, "lon": 79.3334, "alt": 130, "dist": "Ranipet",         "cz": "semi-arid",          "T": 50},
    "Walajapet":         {"lat": 12.9246, "lon": 79.3334, "alt": 130, "dist": "Ranipet",         "cz": "semi-arid",          "T": 50},
    "Sholinghur":        {"lat": 13.1196, "lon": 79.4246, "alt": 100, "dist": "Ranipet",         "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # TIRUPATTUR DISTRICT
    # ════════════════════════════════════════════════════
    "Tirupattur":        {"lat": 12.4961, "lon": 78.5655, "alt": 490, "dist": "Tirupattur",      "cz": "semi-arid",          "T": 50},
    "Ambur":             {"lat": 12.7931, "lon": 78.7151, "alt": 265, "dist": "Tirupattur",      "cz": "semi-arid",          "T": 50},
    "Vaniyambadi":       {"lat": 12.6879, "lon": 78.6186, "alt": 285, "dist": "Tirupattur",      "cz": "semi-arid",          "T": 50},
    "Jolarpettai":       {"lat": 12.5638, "lon": 78.5779, "alt": 380, "dist": "Tirupattur",      "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # TIRUVANNAMALAI DISTRICT
    # ════════════════════════════════════════════════════
    "Tiruvannamalai":    {"lat": 12.2253, "lon": 79.0747, "alt": 188, "dist": "Tiruvannamalai",  "cz": "semi-arid",          "T": 50},
    "Polur":             {"lat": 12.5238, "lon": 79.1288, "alt": 220, "dist": "Tiruvannamalai",  "cz": "semi-arid",          "T": 50},
    "Arani":             {"lat": 12.6648, "lon": 79.2813, "alt": 150, "dist": "Tiruvannamalai",  "cz": "semi-arid",          "T": 50},
    "Cheyyar":           {"lat": 12.6648, "lon": 79.5422, "alt": 100, "dist": "Tiruvannamalai",  "cz": "semi-arid",          "T": 50},
    "Vandavasi":         {"lat": 12.4996, "lon": 79.6295, "alt":  60, "dist": "Tiruvannamalai",  "cz": "semi-arid",          "T": 50},
    "Chetput":           {"lat": 12.7000, "lon": 78.9800, "alt": 200, "dist": "Tiruvannamalai",  "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # KALLAKURICHI DISTRICT
    # ════════════════════════════════════════════════════
    "Kallakurichi":      {"lat": 11.7380, "lon": 78.9598, "alt": 175, "dist": "Kallakurichi",    "cz": "semi-arid",          "T": 50},
    "Ulundurpet":        {"lat": 11.5645, "lon": 79.3161, "alt":  45, "dist": "Kallakurichi",    "cz": "semi-arid",          "T": 50},
    "Sankarapuram":      {"lat": 11.9000, "lon": 78.8800, "alt": 150, "dist": "Kallakurichi",    "cz": "semi-arid",          "T": 50},
    "Rishivandiyam":     {"lat": 11.6200, "lon": 79.1000, "alt": 100, "dist": "Kallakurichi",    "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # VILUPPURAM DISTRICT
    # ════════════════════════════════════════════════════
    "Viluppuram":        {"lat": 11.9401, "lon": 79.4861, "alt":  20, "dist": "Viluppuram",      "cz": "hot-humid",          "T": 45},
    "Tindivanam":        {"lat": 12.2338, "lon": 79.6556, "alt":  25, "dist": "Viluppuram",      "cz": "hot-humid",          "T": 45},
    "Gingee":            {"lat": 12.2527, "lon": 79.4160, "alt":  80, "dist": "Viluppuram",      "cz": "semi-arid",          "T": 50},
    "Tirukoilur":        {"lat": 11.9668, "lon": 79.2000, "alt":  30, "dist": "Viluppuram",      "cz": "semi-arid",          "T": 50},
    "Wandiwash":         {"lat": 12.5028, "lon": 79.6204, "alt":  50, "dist": "Viluppuram",      "cz": "hot-humid",          "T": 45},

    # ════════════════════════════════════════════════════
    # CUDDALORE DISTRICT
    # ════════════════════════════════════════════════════
    "Cuddalore":         {"lat": 11.7447, "lon": 79.7689, "alt":   7, "dist": "Cuddalore",       "cz": "hot-humid-coastal",  "T": 45},
    "Chidambaram":       {"lat": 11.3993, "lon": 79.6920, "alt":   5, "dist": "Cuddalore",       "cz": "hot-humid-coastal",  "T": 45},
    "Panruti":           {"lat": 11.7692, "lon": 79.5655, "alt":  10, "dist": "Cuddalore",       "cz": "hot-humid",          "T": 45},
    "Virudhachalam":     {"lat": 11.5241, "lon": 79.3216, "alt":  20, "dist": "Cuddalore",       "cz": "hot-humid",          "T": 45},
    "Neyveli":           {"lat": 11.5449, "lon": 79.4876, "alt":  15, "dist": "Cuddalore",       "cz": "hot-humid",          "T": 45},
    "Pondicherry":       {"lat": 11.9416, "lon": 79.8083, "alt":  10, "dist": "Puducherry",      "cz": "hot-humid-coastal",  "T": 45},
    "Karaikal":          {"lat": 10.9254, "lon": 79.8380, "alt":   5, "dist": "Karaikal",        "cz": "hot-humid-coastal",  "T": 45},

    # ════════════════════════════════════════════════════
    # DHARMAPURI DISTRICT
    # ════════════════════════════════════════════════════
    "Dharmapuri":        {"lat": 12.1280, "lon": 78.1582, "alt": 389, "dist": "Dharmapuri",      "cz": "semi-arid",          "T": 50},
    "Harur":             {"lat": 12.0506, "lon": 78.4774, "alt": 530, "dist": "Dharmapuri",      "cz": "semi-arid",          "T": 50},
    "Pappireddipatti":   {"lat": 12.2867, "lon": 78.3525, "alt": 350, "dist": "Dharmapuri",      "cz": "semi-arid",          "T": 50},
    "Nallampalli":       {"lat": 12.0500, "lon": 78.0200, "alt": 400, "dist": "Dharmapuri",      "cz": "semi-arid",          "T": 50},
    "Pennagaram":        {"lat": 12.1300, "lon": 77.9200, "alt": 360, "dist": "Dharmapuri",      "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # KRISHNAGIRI DISTRICT
    # ════════════════════════════════════════════════════
    "Krishnagiri":       {"lat": 12.5266, "lon": 78.2139, "alt": 400, "dist": "Krishnagiri",     "cz": "semi-arid",          "T": 50},
    "Hosur":             {"lat": 12.7409, "lon": 77.8253, "alt": 920, "dist": "Krishnagiri",     "cz": "semi-arid-elevated", "T": 55},
    "Denkanikottai":     {"lat": 12.5392, "lon": 77.9799, "alt": 830, "dist": "Krishnagiri",     "cz": "semi-arid-elevated", "T": 55},
    "Bargur":            {"lat": 12.3800, "lon": 78.3500, "alt": 550, "dist": "Krishnagiri",     "cz": "semi-arid",          "T": 50},
    "Shoolagiri":        {"lat": 12.6483, "lon": 78.0583, "alt": 450, "dist": "Krishnagiri",     "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # SALEM DISTRICT
    # ════════════════════════════════════════════════════
    "Salem":             {"lat": 11.6643, "lon": 78.1460, "alt": 278, "dist": "Salem",           "cz": "semi-arid",          "T": 50},
    "Yercaud":           {"lat": 11.7753, "lon": 78.2085, "alt":1515, "dist": "Salem",           "cz": "cool-hilly",         "T": 60},
    "Omalur":            {"lat": 11.7309, "lon": 78.0511, "alt": 250, "dist": "Salem",           "cz": "semi-arid",          "T": 50},
    "Mettur":            {"lat": 11.7905, "lon": 77.7966, "alt": 200, "dist": "Salem",           "cz": "semi-arid",          "T": 50},
    "Edappadi":          {"lat": 11.5866, "lon": 77.9964, "alt": 220, "dist": "Salem",           "cz": "semi-arid",          "T": 50},
    "Attur":             {"lat": 11.5789, "lon": 78.6024, "alt": 240, "dist": "Salem",           "cz": "semi-arid",          "T": 50},
    "Sankari":           {"lat": 11.5140, "lon": 77.8400, "alt": 180, "dist": "Salem",           "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # NAMAKKAL DISTRICT
    # ════════════════════════════════════════════════════
    "Namakkal":          {"lat": 11.2196, "lon": 78.1677, "alt": 183, "dist": "Namakkal",        "cz": "semi-arid",          "T": 50},
    "Tiruchengode":      {"lat": 11.3838, "lon": 77.8948, "alt": 210, "dist": "Namakkal",        "cz": "semi-arid",          "T": 50},
    "Rasipuram":         {"lat": 11.4690, "lon": 78.1800, "alt": 220, "dist": "Namakkal",        "cz": "semi-arid",          "T": 50},
    "Kolli_Hills":       {"lat": 11.2951, "lon": 78.3612, "alt":1300, "dist": "Namakkal",        "cz": "cool-hilly",         "T": 60},
    "Paramathi":         {"lat": 11.3700, "lon": 78.0000, "alt": 200, "dist": "Namakkal",        "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # ERODE DISTRICT
    # ════════════════════════════════════════════════════
    "Erode":             {"lat": 11.3410, "lon": 77.7172, "alt": 158, "dist": "Erode",           "cz": "semi-arid",          "T": 50},
    "Bhavani":           {"lat": 11.4469, "lon": 77.6831, "alt": 150, "dist": "Erode",           "cz": "semi-arid",          "T": 50},
    "Sathyamangalam":    {"lat": 11.5019, "lon": 77.2387, "alt": 270, "dist": "Erode",           "cz": "semi-arid",          "T": 50},
    "Gobichettipalayam": {"lat": 11.4500, "lon": 77.4500, "alt": 190, "dist": "Erode",           "cz": "semi-arid",          "T": 50},
    "Perundurai":        {"lat": 11.2764, "lon": 77.5817, "alt": 170, "dist": "Erode",           "cz": "semi-arid",          "T": 50},
    "Anthiyur":          {"lat": 11.5702, "lon": 77.5872, "alt": 250, "dist": "Erode",           "cz": "semi-arid",          "T": 50},
    "Nambiyur":          {"lat": 11.3870, "lon": 77.1000, "alt": 300, "dist": "Erode",           "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # COIMBATORE DISTRICT
    # ════════════════════════════════════════════════════
    "Coimbatore":        {"lat": 11.0168, "lon": 76.9558, "alt": 411, "dist": "Coimbatore",      "cz": "hot-humid",          "T": 45},
    "Ettimadai":         {"lat": 10.9282, "lon": 76.8780, "alt": 580, "dist": "Coimbatore",      "cz": "hot-humid-elevated", "T": 50},
    "Pollachi":          {"lat": 10.6582, "lon": 77.0080, "alt": 340, "dist": "Coimbatore",      "cz": "hot-humid",          "T": 45},
    "Mettupalayam":      {"lat": 11.2978, "lon": 76.9381, "alt": 300, "dist": "Coimbatore",      "cz": "hot-humid",          "T": 45},
    "Annur":             {"lat": 11.2323, "lon": 77.1022, "alt": 380, "dist": "Coimbatore",      "cz": "hot-humid",          "T": 45},
    "Sulur":             {"lat": 11.0348, "lon": 77.1391, "alt": 395, "dist": "Coimbatore",      "cz": "hot-humid",          "T": 45},
    "Kinathukadavu":     {"lat": 10.8390, "lon": 77.0551, "alt": 440, "dist": "Coimbatore",      "cz": "hot-humid",          "T": 45},
    "Valparai":          {"lat": 10.3260, "lon": 76.9576, "alt":1080, "dist": "Coimbatore",      "cz": "cool-hilly",         "T": 60},
    "Anamalai":          {"lat": 10.3500, "lon": 76.9300, "alt": 900, "dist": "Coimbatore",      "cz": "cool-hilly",         "T": 60},
    "Periyanaickenpalayam": {"lat": 11.1060, "lon": 77.0000, "alt": 400, "dist": "Coimbatore",   "cz": "hot-humid",          "T": 45},

    # ════════════════════════════════════════════════════
    # TIRUPPUR DISTRICT
    # ════════════════════════════════════════════════════
    "Tiruppur":          {"lat": 11.1085, "lon": 77.3411, "alt": 306, "dist": "Tiruppur",        "cz": "semi-arid",          "T": 50},
    "Dharapuram":        {"lat": 10.7341, "lon": 77.5098, "alt": 280, "dist": "Tiruppur",        "cz": "semi-arid",          "T": 50},
    "Palladam":          {"lat": 10.9820, "lon": 77.2846, "alt": 290, "dist": "Tiruppur",        "cz": "semi-arid",          "T": 50},
    "Udumalpet":         {"lat": 10.5847, "lon": 77.2489, "alt": 350, "dist": "Tiruppur",        "cz": "semi-arid",          "T": 50},
    "Avinashi":          {"lat": 11.1937, "lon": 77.2643, "alt": 320, "dist": "Tiruppur",        "cz": "semi-arid",          "T": 50},
    "Kundadam":          {"lat": 10.8600, "lon": 77.4700, "alt": 280, "dist": "Tiruppur",        "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # NILGIRIS DISTRICT
    # ════════════════════════════════════════════════════
    "Ooty":              {"lat": 11.4102, "lon": 76.6950, "alt":2240, "dist": "Nilgiris",        "cz": "cool-hilly",         "T": 65},
    "Coonoor":           {"lat": 11.3530, "lon": 76.7959, "alt":1800, "dist": "Nilgiris",        "cz": "cool-hilly",         "T": 65},
    "Gudalur":           {"lat": 11.5016, "lon": 76.4928, "alt": 950, "dist": "Nilgiris",        "cz": "cool-hilly",         "T": 60},
    "Mudumalai":         {"lat": 11.5667, "lon": 76.6167, "alt":1000, "dist": "Nilgiris",        "cz": "cool-hilly",         "T": 60},
    "Kotagiri":          {"lat": 11.4253, "lon": 76.8633, "alt":1794, "dist": "Nilgiris",        "cz": "cool-hilly",         "T": 65},
    "Pandalur":          {"lat": 11.4600, "lon": 76.4000, "alt":1000, "dist": "Nilgiris",        "cz": "cool-hilly",         "T": 60},

    # ════════════════════════════════════════════════════
    # KARUR DISTRICT
    # ════════════════════════════════════════════════════
    "Karur":             {"lat": 10.9601, "lon": 78.0766, "alt": 122, "dist": "Karur",           "cz": "semi-arid",          "T": 50},
    "Kulithalai":        {"lat": 10.9333, "lon": 78.4247, "alt":  80, "dist": "Karur",           "cz": "semi-arid",          "T": 50},
    "Aravakurichi":      {"lat": 10.9100, "lon": 78.1800, "alt": 120, "dist": "Karur",           "cz": "semi-arid",          "T": 50},
    "Manmangalam":       {"lat": 10.8800, "lon": 78.2300, "alt": 100, "dist": "Karur",           "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # TIRUCHIRAPPALLI DISTRICT
    # ════════════════════════════════════════════════════
    "Tiruchirappalli":   {"lat": 10.7905, "lon": 78.7047, "alt":  88, "dist": "Tiruchirappalli", "cz": "semi-arid",          "T": 50},
    "Srirangam":         {"lat": 10.8630, "lon": 78.6896, "alt":  78, "dist": "Tiruchirappalli", "cz": "semi-arid",          "T": 50},
    "Lalgudi":           {"lat": 10.8740, "lon": 78.8204, "alt":  70, "dist": "Tiruchirappalli", "cz": "semi-arid",          "T": 50},
    "Musiri":            {"lat": 10.9494, "lon": 78.4366, "alt":  90, "dist": "Tiruchirappalli", "cz": "semi-arid",          "T": 50},
    "Thottiyam":         {"lat": 11.0000, "lon": 78.3000, "alt": 100, "dist": "Tiruchirappalli", "cz": "semi-arid",          "T": 50},
    "Manachanallur":     {"lat": 10.8600, "lon": 78.7500, "alt":  80, "dist": "Tiruchirappalli", "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # PERAMBALUR DISTRICT
    # ════════════════════════════════════════════════════
    "Perambalur":        {"lat": 11.2334, "lon": 78.8803, "alt":  78, "dist": "Perambalur",      "cz": "semi-arid",          "T": 50},
    "Kunnam":            {"lat": 11.3400, "lon": 79.0100, "alt":  70, "dist": "Perambalur",      "cz": "semi-arid",          "T": 50},
    "Veppanthattai":     {"lat": 11.3100, "lon": 78.8200, "alt":  80, "dist": "Perambalur",      "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # ARIYALUR DISTRICT
    # ════════════════════════════════════════════════════
    "Ariyalur":          {"lat": 11.1387, "lon": 79.0783, "alt":  68, "dist": "Ariyalur",        "cz": "semi-arid",          "T": 50},
    "Udayarpalayam":     {"lat": 11.2200, "lon": 79.1700, "alt":  60, "dist": "Ariyalur",        "cz": "semi-arid",          "T": 50},
    "Sendurai":          {"lat": 11.1500, "lon": 79.2500, "alt":  50, "dist": "Ariyalur",        "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # THANJAVUR DISTRICT
    # ════════════════════════════════════════════════════
    "Thanjavur":         {"lat": 10.7870, "lon": 79.1378, "alt":  57, "dist": "Thanjavur",       "cz": "hot-humid",          "T": 45},
    "Kumbakonam":        {"lat": 10.9617, "lon": 79.3845, "alt":  28, "dist": "Thanjavur",       "cz": "hot-humid",          "T": 45},
    "Papanasam":         {"lat": 10.9328, "lon": 79.2737, "alt":  30, "dist": "Thanjavur",       "cz": "hot-humid",          "T": 45},
    "Pattukkottai":      {"lat": 10.4286, "lon": 79.3191, "alt":   5, "dist": "Thanjavur",       "cz": "hot-humid-coastal",  "T": 45},
    "Pattukottai":       {"lat": 10.4286, "lon": 79.3191, "alt":   5, "dist": "Thanjavur",       "cz": "hot-humid",          "T": 45},
    "Thiruvaiyaru":      {"lat": 10.8827, "lon": 79.0967, "alt":  45, "dist": "Thanjavur",       "cz": "hot-humid",          "T": 45},
    "Budalur":           {"lat": 10.7200, "lon": 79.1600, "alt":  50, "dist": "Thanjavur",       "cz": "hot-humid",          "T": 45},
    "Orathanadu":        {"lat": 10.5800, "lon": 79.2600, "alt":  20, "dist": "Thanjavur",       "cz": "hot-humid",          "T": 45},

    # ════════════════════════════════════════════════════
    # MAYILADUTHURAI DISTRICT
    # ════════════════════════════════════════════════════
    "Mayiladuthurai":    {"lat": 11.1016, "lon": 79.6519, "alt":  10, "dist": "Mayiladuthurai",  "cz": "hot-humid",          "T": 45},
    "Sirkazhi":          {"lat": 11.2317, "lon": 79.7386, "alt":   5, "dist": "Mayiladuthurai",  "cz": "hot-humid-coastal",  "T": 45},
    "Tarangambadi":      {"lat": 11.0153, "lon": 79.8455, "alt":   3, "dist": "Mayiladuthurai",  "cz": "hot-humid-coastal",  "T": 45},
    "Kuthalam":          {"lat": 11.0700, "lon": 79.7100, "alt":   5, "dist": "Mayiladuthurai",  "cz": "hot-humid",          "T": 45},

    # ════════════════════════════════════════════════════
    # NAGAPATTINAM DISTRICT
    # ════════════════════════════════════════════════════
    "Nagapattinam":      {"lat": 10.7672, "lon": 79.8449, "alt":   3, "dist": "Nagapattinam",    "cz": "hot-humid-coastal",  "T": 45},
    "Vedaranyam":        {"lat": 10.3773, "lon": 79.8505, "alt":   3, "dist": "Nagapattinam",    "cz": "hot-humid-coastal",  "T": 45},
    "Kilvelur":          {"lat": 10.7000, "lon": 79.8000, "alt":   5, "dist": "Nagapattinam",    "cz": "hot-humid-coastal",  "T": 45},
    "Tharangambadi":     {"lat": 11.0200, "lon": 79.8500, "alt":   3, "dist": "Nagapattinam",    "cz": "hot-humid-coastal",  "T": 45},
    "Velankanni":        {"lat": 10.6852, "lon": 79.8596, "alt":   5, "dist": "Nagapattinam",    "cz": "hot-humid-coastal",  "T": 45},

    # ════════════════════════════════════════════════════
    # TIRUVARUR DISTRICT
    # ════════════════════════════════════════════════════
    "Tiruvarur":         {"lat": 10.7726, "lon": 79.6366, "alt":   8, "dist": "Tiruvarur",       "cz": "hot-humid",          "T": 45},
    "Mannargudi":        {"lat": 10.6640, "lon": 79.4536, "alt":  10, "dist": "Tiruvarur",       "cz": "hot-humid",          "T": 45},
    "Thiruthuraipoondi": {"lat": 10.5282, "lon": 79.6434, "alt":   5, "dist": "Tiruvarur",       "cz": "hot-humid-coastal",  "T": 45},
    "Papanasam_TV":      {"lat": 10.9328, "lon": 79.2737, "alt":  30, "dist": "Tiruvarur",       "cz": "hot-humid",          "T": 45},

    # ════════════════════════════════════════════════════
    # PUDUKKOTTAI DISTRICT
    # ════════════════════════════════════════════════════
    "Pudukkottai":       {"lat": 10.3797, "lon": 78.8198, "alt": 100, "dist": "Pudukkottai",     "cz": "semi-arid",          "T": 50},
    "Aranthangi":        {"lat": 10.1672, "lon": 79.0789, "alt":  20, "dist": "Pudukkottai",     "cz": "semi-arid",          "T": 50},
    "Karambakkudi":      {"lat": 10.4500, "lon": 78.8800, "alt":  80, "dist": "Pudukkottai",     "cz": "semi-arid",          "T": 50},
    "Alangudi":          {"lat": 10.4100, "lon": 79.1300, "alt":  30, "dist": "Pudukkottai",     "cz": "semi-arid",          "T": 50},
    "Gandharvakottai":   {"lat": 10.5500, "lon": 78.9700, "alt":  60, "dist": "Pudukkottai",     "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # DINDIGUL DISTRICT
    # ════════════════════════════════════════════════════
    "Dindigul":          {"lat": 10.3624, "lon": 77.9695, "alt": 283, "dist": "Dindigul",        "cz": "semi-arid",          "T": 50},
    "Kodaikanal":        {"lat": 10.2381, "lon": 77.4892, "alt":2133, "dist": "Dindigul",        "cz": "cool-hilly",         "T": 65},
    "Palani":            {"lat": 10.4494, "lon": 77.5244, "alt": 350, "dist": "Dindigul",        "cz": "semi-arid",          "T": 50},
    "Vedasandur":        {"lat": 10.5310, "lon": 77.9547, "alt": 290, "dist": "Dindigul",        "cz": "semi-arid",          "T": 50},
    "Natham":            {"lat": 10.3300, "lon": 78.1700, "alt": 250, "dist": "Dindigul",        "cz": "semi-arid",          "T": 50},
    "Oddanchatram":      {"lat": 10.3094, "lon": 77.7472, "alt": 310, "dist": "Dindigul",        "cz": "semi-arid",          "T": 50},
    "Nilakkottai":       {"lat": 10.1700, "lon": 77.8600, "alt": 280, "dist": "Dindigul",        "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # THENI DISTRICT
    # ════════════════════════════════════════════════════
    "Theni":             {"lat": 10.0104, "lon": 77.4766, "alt": 289, "dist": "Theni",           "cz": "semi-arid",          "T": 50},
    "Bodinayakanur":     {"lat": 10.0107, "lon": 77.3526, "alt": 490, "dist": "Theni",           "cz": "semi-arid",          "T": 50},
    "Periyakulam":       {"lat": 10.1166, "lon": 77.5481, "alt": 335, "dist": "Theni",           "cz": "semi-arid",          "T": 50},
    "Andipatti":         {"lat":  9.9900, "lon": 77.6200, "alt": 270, "dist": "Theni",           "cz": "semi-arid",          "T": 50},
    "Uthamapalayam":     {"lat": 10.0600, "lon": 77.3200, "alt": 400, "dist": "Theni",           "cz": "semi-arid",          "T": 50},

    # ════════════════════════════════════════════════════
    # MADURAI DISTRICT
    # ════════════════════════════════════════════════════
    "Madurai":           {"lat":  9.9252, "lon": 78.1198, "alt": 101, "dist": "Madurai",         "cz": "hot-semi-arid",      "T": 50},
    "Melur":             {"lat":  9.9700, "lon": 78.3300, "alt":  90, "dist": "Madurai",         "cz": "hot-semi-arid",      "T": 50},
    "Usilampatti":       {"lat":  9.9681, "lon": 77.8007, "alt": 120, "dist": "Madurai",         "cz": "hot-semi-arid",      "T": 50},
    "Tirumangalam":      {"lat":  9.8215, "lon": 77.9804, "alt":  80, "dist": "Madurai",         "cz": "hot-semi-arid",      "T": 50},
    "Vadipatti":         {"lat":  9.9900, "lon": 78.0000, "alt": 110, "dist": "Madurai",         "cz": "hot-semi-arid",      "T": 50},
    "Peraiyur":          {"lat":  9.7500, "lon": 78.2300, "alt":  80, "dist": "Madurai",         "cz": "hot-semi-arid",      "T": 50},

    # ════════════════════════════════════════════════════
    # SIVAGANGA DISTRICT
    # ════════════════════════════════════════════════════
    "Sivaganga":         {"lat":  9.8479, "lon": 78.4808, "alt":  90, "dist": "Sivaganga",       "cz": "hot-semi-arid",      "T": 50},
    "Karaikudi":         {"lat": 10.0767, "lon": 78.7832, "alt":  90, "dist": "Sivaganga",       "cz": "hot-semi-arid",      "T": 50},
    "Devakottai":        {"lat":  9.9577, "lon": 78.8246, "alt":  80, "dist": "Sivaganga",       "cz": "hot-semi-arid",      "T": 50},
    "Manamadurai":       {"lat":  9.6783, "lon": 78.4643, "alt":  70, "dist": "Sivaganga",       "cz": "hot-semi-arid",      "T": 50},
    "Tiruppathur_SG":    {"lat":  9.9000, "lon": 78.5700, "alt":  80, "dist": "Sivaganga",       "cz": "hot-semi-arid",      "T": 50},

    # ════════════════════════════════════════════════════
    # RAMANATHAPURAM DISTRICT
    # ════════════════════════════════════════════════════
    "Ramanathapuram":    {"lat":  9.3762, "lon": 78.8308, "alt":   9, "dist": "Ramanathapuram",  "cz": "hot-arid-coastal",   "T": 50},
    "Rameswaram":        {"lat":  9.2881, "lon": 79.3129, "alt":   5, "dist": "Ramanathapuram",  "cz": "hot-arid-coastal",   "T": 50},
    "Paramakudi":        {"lat":  9.5303, "lon": 78.5957, "alt":  30, "dist": "Ramanathapuram",  "cz": "hot-arid",           "T": 50},
    "Kilakkarai":        {"lat":  9.2311, "lon": 78.8003, "alt":   5, "dist": "Ramanathapuram",  "cz": "hot-arid-coastal",   "T": 50},
    "Tondi":             {"lat":  9.7498, "lon": 79.0154, "alt":   3, "dist": "Ramanathapuram",  "cz": "hot-arid-coastal",   "T": 50},
    "Tiruvadanai":       {"lat":  9.7500, "lon": 79.0400, "alt":   5, "dist": "Ramanathapuram",  "cz": "hot-arid-coastal",   "T": 50},
    "Mandapam":          {"lat":  9.2753, "lon": 79.1282, "alt":   3, "dist": "Ramanathapuram",  "cz": "hot-arid-coastal",   "T": 50},
    "Dhanushkodi":       {"lat":  9.1788, "lon": 79.4176, "alt":   2, "dist": "Ramanathapuram",  "cz": "hot-arid-coastal",   "T": 50},

    # ════════════════════════════════════════════════════
    # VIRUDHUNAGAR DISTRICT
    # ════════════════════════════════════════════════════
    "Virudhunagar":      {"lat":  9.5850, "lon": 77.9624, "alt":  96, "dist": "Virudhunagar",    "cz": "hot-semi-arid",      "T": 50},
    "Sivakasi":          {"lat":  9.4521, "lon": 77.7980, "alt":  90, "dist": "Virudhunagar",    "cz": "hot-semi-arid",      "T": 50},
    "Sattur":            {"lat":  9.3483, "lon": 77.9082, "alt":  60, "dist": "Virudhunagar",    "cz": "hot-semi-arid",      "T": 50},
    "Rajapalayam":       {"lat":  9.4542, "lon": 77.5561, "alt": 120, "dist": "Virudhunagar",    "cz": "hot-semi-arid",      "T": 50},
    "Srivilliputhur":    {"lat":  9.5103, "lon": 77.6347, "alt": 100, "dist": "Virudhunagar",    "cz": "hot-semi-arid",      "T": 50},
    "Aruppukkottai":     {"lat":  9.5082, "lon": 78.0980, "alt":  70, "dist": "Virudhunagar",    "cz": "hot-semi-arid",      "T": 50},
    "Tiruchuli":         {"lat":  9.2800, "lon": 77.7700, "alt":  60, "dist": "Virudhunagar",    "cz": "hot-semi-arid",      "T": 50},
    "Watrap":            {"lat":  9.4100, "lon": 77.5200, "alt": 110, "dist": "Virudhunagar",    "cz": "hot-semi-arid",      "T": 50},

    # ════════════════════════════════════════════════════
    # THOOTHUKUDI DISTRICT
    # ════════════════════════════════════════════════════
    "Thoothukudi":       {"lat":  8.7642, "lon": 78.1348, "alt":  11, "dist": "Thoothukudi",     "cz": "hot-arid-coastal",   "T": 50},
    "Kovilpatti":        {"lat":  9.1750, "lon": 77.8688, "alt":  68, "dist": "Thoothukudi",     "cz": "hot-semi-arid",      "T": 50},
    "Ettayapuram":       {"lat":  9.0975, "lon": 78.0010, "alt":  25, "dist": "Thoothukudi",     "cz": "hot-semi-arid",      "T": 50},
    "Tiruchendur":       {"lat":  8.4946, "lon": 78.1195, "alt":   5, "dist": "Thoothukudi",     "cz": "hot-arid-coastal",   "T": 50},
    "Kayalpatnam":       {"lat":  8.5668, "lon": 78.1145, "alt":   3, "dist": "Thoothukudi",     "cz": "hot-arid-coastal",   "T": 50},
    "Srivaikuntam":      {"lat":  8.6282, "lon": 77.9148, "alt":  10, "dist": "Thoothukudi",     "cz": "hot-arid",           "T": 50},

    # ════════════════════════════════════════════════════
    # TIRUNELVELI DISTRICT
    # ════════════════════════════════════════════════════
    "Tirunelveli":       {"lat":  8.7139, "lon": 77.7567, "alt":  45, "dist": "Tirunelveli",     "cz": "hot-semi-arid",      "T": 50},
    "Palayamkottai":     {"lat":  8.7268, "lon": 77.7372, "alt":  40, "dist": "Tirunelveli",     "cz": "hot-semi-arid",      "T": 50},
    "Nanguneri":         {"lat":  8.4874, "lon": 77.6481, "alt":  30, "dist": "Tirunelveli",     "cz": "hot-humid",          "T": 45},
    "Cheranmahadevi":    {"lat":  8.6944, "lon": 77.5789, "alt":  50, "dist": "Tirunelveli",     "cz": "hot-semi-arid",      "T": 50},
    "Ambasamudram":      {"lat":  8.7100, "lon": 77.4600, "alt":  80, "dist": "Tirunelveli",     "cz": "hot-humid",          "T": 45},
    "Manur":             {"lat":  8.8200, "lon": 77.5300, "alt":  60, "dist": "Tirunelveli",     "cz": "hot-semi-arid",      "T": 50},
    "Radhapuram":        {"lat":  8.3547, "lon": 77.6258, "alt":  15, "dist": "Tirunelveli",     "cz": "hot-humid-coastal",  "T": 45},

    # ════════════════════════════════════════════════════
    # TENKASI DISTRICT
    # ════════════════════════════════════════════════════
    "Tenkasi":           {"lat":  8.9592, "lon": 77.3153, "alt":  93, "dist": "Tenkasi",         "cz": "hot-humid",          "T": 45},
    "Sankarankovil":     {"lat":  9.1710, "lon": 77.5474, "alt": 100, "dist": "Tenkasi",         "cz": "hot-semi-arid",      "T": 50},
    "Shenkottai":        {"lat":  8.9738, "lon": 77.2501, "alt": 120, "dist": "Tenkasi",         "cz": "hot-humid",          "T": 45},
    "Alangulam":         {"lat":  8.8600, "lon": 77.4200, "alt":  70, "dist": "Tenkasi",         "cz": "hot-semi-arid",      "T": 50},
    "Kadayanallur":      {"lat":  9.0750, "lon": 77.3400, "alt":  90, "dist": "Tenkasi",         "cz": "hot-humid",          "T": 45},
    "Courtallam":        {"lat":  8.9278, "lon": 77.2803, "alt": 160, "dist": "Tenkasi",         "cz": "hot-humid",          "T": 45},

    # ════════════════════════════════════════════════════
    # KANYAKUMARI DISTRICT
    # ════════════════════════════════════════════════════
    "Kanyakumari":       {"lat":  8.0883, "lon": 77.5385, "alt":   8, "dist": "Kanyakumari",     "cz": "hot-humid-coastal",  "T": 45},
    "Nagercoil":         {"lat":  8.1833, "lon": 77.4119, "alt":  10, "dist": "Kanyakumari",     "cz": "hot-humid-coastal",  "T": 45},
    "Padmanabhapuram":   {"lat":  8.2513, "lon": 77.3376, "alt":  15, "dist": "Kanyakumari",     "cz": "hot-humid",          "T": 45},
    "Thuckalay":         {"lat":  8.2432, "lon": 77.3027, "alt":  20, "dist": "Kanyakumari",     "cz": "hot-humid",          "T": 45},
    "Marthandam":        {"lat":  8.3116, "lon": 77.2327, "alt":  30, "dist": "Kanyakumari",     "cz": "hot-humid",          "T": 45},
    "Colachel":          {"lat":  8.1775, "lon": 77.2653, "alt":   5, "dist": "Kanyakumari",     "cz": "hot-humid-coastal",  "T": 45},
    "Kuzhithurai":       {"lat":  8.3153, "lon": 77.1872, "alt":  15, "dist": "Kanyakumari",     "cz": "hot-humid",          "T": 45},
    "Killiyoor":         {"lat":  8.1700, "lon": 77.4200, "alt":  10, "dist": "Kanyakumari",     "cz": "hot-humid-coastal",  "T": 45},

    # ════════════════════════════════════════════════════
    # ADDITIONAL IMPORTANT TOWNS / COASTAL / TOURISM
    # ════════════════════════════════════════════════════
    "Adirampattinam":    {"lat": 10.3551, "lon": 79.3832, "alt":   3, "dist": "Thanjavur",       "cz": "hot-humid-coastal",  "T": 45},
    "Nagore":            {"lat": 10.8293, "lon": 79.8442, "alt":   3, "dist": "Nagapattinam",    "cz": "hot-humid-coastal",  "T": 45},
    "Kodiyakarai":       {"lat": 10.3397, "lon": 79.8833, "alt":   2, "dist": "Nagapattinam",    "cz": "hot-humid-coastal",  "T": 45},
    "Cuddalore_Port":    {"lat": 11.7600, "lon": 79.7800, "alt":   3, "dist": "Cuddalore",       "cz": "hot-humid-coastal",  "T": 45},
    "Tambaram_South":    {"lat": 12.8800, "lon": 80.0600, "alt":  30, "dist": "Chengalpattu",    "cz": "hot-humid",          "T": 45},
    "Tirumayam":         {"lat": 10.2600, "lon": 78.7400, "alt": 110, "dist": "Pudukkottai",     "cz": "semi-arid",          "T": 50},
    "Ilayangudi":        {"lat":  9.7200, "lon": 78.6100, "alt":  60, "dist": "Sivaganga",       "cz": "hot-semi-arid",      "T": 50},
    "Peravurani":        {"lat": 10.2800, "lon": 79.2200, "alt":   8, "dist": "Thanjavur",       "cz": "hot-humid",          "T": 45},
}


# ═══════════════════════════════════════════════════════════
# FILE OPENING (Python 3.14 safe)
# ═══════════════════════════════════════════════════════════

def open_nc(fpath):
    """Try multiple engines; fall back to mask_and_scale=False for Python 3.14."""
    if not os.path.exists(fpath):
        return None
    for engine in ("netcdf4", "scipy", "h5netcdf"):
        try:
            return xr.open_dataset(fpath, engine=engine)
        except Exception:
            pass
    try:
        return xr.open_dataset(
            fpath, engine="netcdf4",
            mask_and_scale=False, decode_cf=False, decode_times=False)
    except Exception as e:
        print(f"    [WARN] Cannot open {os.path.basename(fpath)}: {e}")
        return None


def safe_values(da):
    """Extract float array, apply CF scale/offset/fill if decode_cf=False was used."""
    arr = da.values.astype(float)
    attrs = getattr(da, "attrs", {})
    if "scale_factor" in attrs:
        arr = arr * float(attrs["scale_factor"])
    if "add_offset" in attrs:
        arr = arr + float(attrs["add_offset"])
    if "_FillValue" in attrs:
        fv = float(attrs["_FillValue"])
        arr = np.where(np.abs(arr - fv) < 1e-4, np.nan, arr)
    return arr


def decode_time(ds_pt):
    """Decode time coordinate to pandas DatetimeIndex (UTC, tz-naive)."""
    for tc in ("valid_time", "time"):
        if tc not in ds_pt.coords and tc not in ds_pt.dims:
            continue
        tv = ds_pt[tc].values
        try:
            t = pd.to_datetime(tv)
            return t.tz_localize(None) if t.tz is not None else t
        except Exception:
            pass
        try:
            units = ds_pt[tc].attrs.get("units", "hours since 1900-01-01")
            t = pd.to_datetime(
                xr.coding.times.decode_cf_datetime(tv, units), utc=True
            ).tz_localize(None)
            return t
        except Exception:
            pass
    return None


def extract_nearest(ds, lat, lon):
    """Extract hourly DataFrame at the nearest ERA5 grid point to (lat, lon)."""
    lat_name = next((c for c in list(ds.coords) + list(ds.dims)
                     if "lat" in c.lower()), None)
    lon_name = next((c for c in list(ds.coords) + list(ds.dims)
                     if "lon" in c.lower()), None)
    if not lat_name or not lon_name:
        return None

    lat_arr = ds[lat_name].values.astype(float)
    lon_arr = ds[lon_name].values.astype(float)
    li = int(np.argmin(np.abs(lat_arr - lat)))
    lo = int(np.argmin(np.abs(lon_arr - lon)))

    ds_pt = ds.isel({lat_name: li, lon_name: lo})
    time_coord = decode_time(ds_pt)
    if time_coord is None:
        return None

    records = {}
    for var in ds_pt.data_vars:
        try:
            arr = safe_values(ds_pt[var]).squeeze()
            if arr.shape == time_coord.shape:
                records[var] = arr
        except Exception:
            pass

    if not records:
        return None

    df = pd.DataFrame(records, index=time_coord)
    df.index.name = "time"
    # Store the actual grid point lat/lon used (nearest ERA5 point)
    df.attrs["grid_lat"] = float(lat_arr[li])
    df.attrs["grid_lon"] = float(lon_arr[lo])
    return df.sort_index()


def extract_grid_point(ds, lat_idx, lon_idx, lat_val, lon_val):
    """Extract hourly DataFrame at a specific grid index."""
    lat_name = next((c for c in list(ds.coords) + list(ds.dims)
                     if "lat" in c.lower()), None)
    lon_name = next((c for c in list(ds.coords) + list(ds.dims)
                     if "lon" in c.lower()), None)
    if not lat_name or not lon_name:
        return None

    ds_pt = ds.isel({lat_name: lat_idx, lon_name: lon_idx})
    time_coord = decode_time(ds_pt)
    if time_coord is None:
        return None

    records = {}
    for var in ds_pt.data_vars:
        try:
            arr = safe_values(ds_pt[var]).squeeze()
            if arr.shape == time_coord.shape:
                records[var] = arr
        except Exception:
            pass

    if not records:
        return None

    df = pd.DataFrame(records, index=time_coord)
    df.index.name = "time"
    return df.sort_index()


# ═══════════════════════════════════════════════════════════
# PHYSICS FUNCTIONS
# ═══════════════════════════════════════════════════════════

def kelvin_to_c(arr):
    return np.asarray(arr, dtype=float) - 273.15


def compute_rh(T_c, Td_c):
    a, b = 17.625, 243.04
    return np.clip(
        100 * np.exp(a * Td_c / (b + Td_c)) / np.exp(a * T_c / (b + T_c)),
        0, 100)


def deaccumulate(s):
    """
    ERA5 hourly reanalysis: accumulated values reset every 12 h.
    Resets happen at hours 1 and 13 UTC (start of each forecast run).
    diff() gives hourly increments; at reset hours the raw value is the increment.
    """
    s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
    diff = s.diff()
    reset_mask = s.index.hour.isin([1, 13])
    diff[reset_mask] = s[reset_mask]
    return diff.clip(lower=0)


def compute_solar(df, lat, lon, alt):
    loc = pvlib.location.Location(latitude=lat, longitude=lon,
                                  altitude=alt, tz="UTC")
    times = pd.DatetimeIndex(df.index)
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen")

    df["SZA"]           = sp["zenith"].values
    df["solar_azimuth"] = sp["azimuth"].values
    df["ETR"]           = pvlib.irradiance.get_extra_radiation(times).values
    df["GHI_clearsky"]  = cs["ghi"].values

    if "GHI" in df.columns:
        df["CSI"] = np.where(
            df["GHI_clearsky"] > 10,
            (df["GHI"] / df["GHI_clearsky"]).clip(0, 1.5), 0)
        cos_z = np.cos(np.deg2rad(df["SZA"].clip(0, 89.9)))
        if "avg_sdirswrf" in df.columns:
            df["DNI"] = df["avg_sdirswrf"].clip(0, 1400)
        else:
            df["DNI"] = np.where(cos_z > 0.05,
                                 df["GHI"] / cos_z, 0).clip(0, 1400)
        df["DHI"] = (df["GHI"] - df["DNI"] * cos_z).clip(0)
    return df


def compute_sunrise_sunset(df, lat, lon, alt):
    loc = pvlib.location.Location(latitude=lat, longitude=lon,
                                  altitude=alt, tz="UTC")
    sr_map, ss_map = {}, {}
    for d in sorted(set(df.index.date)):
        try:
            st = loc.get_sun_rise_set_transit(
                pd.DatetimeIndex([pd.Timestamp(d)]), model="spa")
            sr = st["sunrise"].iloc[0]
            ss = st["sunset"].iloc[0]
            sr_map[d] = sr.hour + sr.minute / 60 if pd.notna(sr) else np.nan
            ss_map[d] = ss.hour + ss.minute / 60 if pd.notna(ss) else np.nan
        except Exception:
            sr_map[d] = np.nan
            ss_map[d] = np.nan
    df["sunrise_hour"] = [sr_map.get(t.date(), np.nan) for t in df.index]
    df["sunset_hour"]  = [ss_map.get(t.date(), np.nan) for t in df.index]
    return df


def compute_rrtdhs(df, T_set):
    vals = []
    for (yr, mo), grp in df.groupby([df.index.year, df.index.month]):
        ghi_m  = grp["GHI"].mean()  if "GHI"   in grp.columns else 0
        t_fit  = float(np.clip(1 - abs(grp["T_amb"].mean() - T_set) / 30, 0, 1)) \
                 if "T_amb" in grp.columns else 0.5
        d_frac = (grp["GHI"] > 10).sum() / max(len(grp), 1) \
                 if "GHI" in grp.columns else 0
        vals.extend([(ghi_m / 1000) * t_fit * (1 + d_frac)] * len(grp))
    df["RRTDHS"] = vals[:len(df)]
    return df


# ═══════════════════════════════════════════════════════════
# UNIT CONVERSION (shared by named-location and grid modes)
# ═══════════════════════════════════════════════════════════

def apply_unit_conversions(df):
    """Convert raw ERA5 variable names to physical columns."""
    # Instant
    if "t2m" in df.columns:
        df["T_amb"] = kelvin_to_c(df["t2m"])
    if "d2m" in df.columns:
        df["T_dew"] = kelvin_to_c(df["d2m"])
    if "T_amb" in df.columns and "T_dew" in df.columns:
        df["RHum"] = compute_rh(df["T_amb"], df["T_dew"])
    if "u10" in df.columns and "v10" in df.columns:
        u = df["u10"].astype(float)
        v = df["v10"].astype(float)
        df["W_spd"] = np.sqrt(u**2 + v**2)
        df["W_dir"] = (np.degrees(np.arctan2(u, v)) + 360) % 360
    if "sp" in df.columns:
        df["P_atm"] = df["sp"].astype(float) / 100.0
    if "tcc" in df.columns:
        df["cloud_cover"] = df["tcc"].astype(float)

    # Accumulated solar / precipitation
    ssrd_col  = next((c for c in df.columns if c == "ssrd"),  None)
    fdir_col  = next((c for c in df.columns
                      if c in ("msdwswrf", "fdir", "msdrswrf")), None)
    strd_col  = next((c for c in df.columns if c == "strd"),  None)
    tp_col    = next((c for c in df.columns if c == "tp"),    None)

    if ssrd_col:
        df["GHI"]         = (deaccumulate(df[ssrd_col].astype(float)) / 3600).clip(0)
    if fdir_col:
        df["avg_sdirswrf"] = df[fdir_col].astype(float).clip(0)
    if strd_col:
        df["LW_down"]     = (deaccumulate(df[strd_col].astype(float)) / 3600).clip(0)
    if tp_col:
        df["precipitation"] = (deaccumulate(df[tp_col].astype(float)) * 1000).clip(0)

    return df


KEEP_COLS = {
    "timestamp", "avg_sdirswrf", "T_amb", "T_dew", "RHum", "W_dir", "W_spd",
    "GHI", "LW_down", "cloud_cover", "precipitation", "P_atm",
    "hour", "month", "DOY", "year", "season", "season_code",
    "SZA", "solar_azimuth", "ETR", "GHI_clearsky", "CSI", "RRTDHS",
    "sunrise_hour", "sunset_hour",
    "city", "grid_lat", "grid_lon", "lat", "lon", "altitude_m",
    "district", "climate_zone", "T_set", "high_solar_resource", "DNI", "DHI",
}

COL_ORDER = [
    "timestamp", "avg_sdirswrf", "T_amb", "T_dew", "RHum", "W_dir", "W_spd",
    "GHI", "LW_down", "cloud_cover", "precipitation", "P_atm",
    "hour", "month", "DOY", "year", "season", "season_code",
    "SZA", "solar_azimuth", "ETR", "GHI_clearsky", "CSI", "RRTDHS",
    "sunrise_hour", "sunset_hour",
    "city", "grid_lat", "grid_lon", "lat", "lon", "altitude_m",
    "district", "climate_zone", "T_set", "high_solar_resource", "DNI", "DHI",
]


def finalize_df(df, lat, lon, alt, T_set, city, district, cz,
                grid_lat=None, grid_lon=None):
    """Add time features, metadata, bounds check, and column ordering."""
    df["timestamp"]   = df.index
    df["hour"]        = df.index.hour
    df["month"]       = df.index.month
    df["DOY"]         = df.index.dayofyear
    df["year"]        = df.index.year
    df["season"]      = df["month"].map(lambda m: SEASON_MAP[m][0])
    df["season_code"] = df["month"].map(lambda m: SEASON_MAP[m][1])

    df = compute_rrtdhs(df, T_set)

    # Physical bounds
    if "GHI" in df.columns:
        df.loc[df["GHI"] < 0,    "GHI"] = 0
        df.loc[df["GHI"] > 1400, "GHI"] = np.nan
    if "T_amb" in df.columns:
        df.loc[df["T_amb"] < -5, "T_amb"] = np.nan
        df.loc[df["T_amb"] > 60, "T_amb"] = np.nan
    if "RHum" in df.columns:
        df["RHum"] = df["RHum"].clip(0, 100)

    # Metadata
    df["city"]               = city
    df["lat"]                = lat
    df["lon"]                = lon
    df["grid_lat"]           = grid_lat if grid_lat is not None else lat
    df["grid_lon"]           = grid_lon if grid_lon is not None else lon
    df["altitude_m"]         = alt
    df["district"]           = district
    df["climate_zone"]       = cz
    df["T_set"]              = T_set
    df["high_solar_resource"] = (
        df.get("RRTDHS", pd.Series(0, index=df.index)) > RRTDHS_THRESHOLD
    ).astype(int)

    # Drop raw ERA5 columns; keep only derived ones
    df.drop(columns=[c for c in df.columns if c not in KEEP_COLS],
            errors="ignore", inplace=True)

    present = [c for c in COL_ORDER if c in df.columns]
    extra   = [c for c in df.columns if c not in COL_ORDER]
    return df[present + extra].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# LOAD ALL NETCDF FILES INTO MEMORY  (open once, reuse)
# ═══════════════════════════════════════════════════════════

def load_all_files():
    """
    Open all available instant and accum NetCDF files.
    Returns two dicts: instant_ds[year][month], accum_ds[year][month]
    """
    print("  Loading NetCDF files into memory ...")
    instant_ds = {}
    accum_ds   = {}

    for year in YEARS:
        instant_ds[year] = {}
        accum_ds[year]   = {}
        for month in MONTHS:
            fi = os.path.join(INPUT_DIR, f"era5_TN_grid_{year}_{month}_instant.nc")
            fa = os.path.join(INPUT_DIR, f"era5_TN_grid_{year}_{month}_accum.nc")
            if os.path.exists(fi):
                ds = open_nc(fi)
                if ds is not None:
                    instant_ds[year][month] = ds
            if os.path.exists(fa):
                ds = open_nc(fa)
                if ds is not None:
                    accum_ds[year][month] = ds

    ni = sum(len(v) for v in instant_ds.values())
    na = sum(len(v) for v in accum_ds.values())
    expected = len(YEARS) * len(MONTHS)
    print(f"  Opened: instant={ni}/{expected}  accum={na}/{expected}")
    return instant_ds, accum_ds


# ═══════════════════════════════════════════════════════════
# PROCESS ONE NAMED LOCATION
# ═══════════════════════════════════════════════════════════

def process_location(name, cfg, instant_ds, accum_ds):
    lat, lon, alt = cfg["lat"], cfg["lon"], cfg["alt"]
    T_set = cfg["T"]

    fi_frames, fa_frames = [], []
    grid_lat_used = grid_lon_used = None

    for year in YEARS:
        for month in MONTHS:
            if month in instant_ds.get(year, {}):
                df_i = extract_nearest(instant_ds[year][month], lat, lon)
                if df_i is not None and len(df_i) > 0:
                    if grid_lat_used is None:
                        grid_lat_used = df_i.attrs.get("grid_lat")
                        grid_lon_used = df_i.attrs.get("grid_lon")
                    fi_frames.append(df_i)

            if month in accum_ds.get(year, {}):
                df_a = extract_nearest(accum_ds[year][month], lat, lon)
                if df_a is not None and len(df_a) > 0:
                    fa_frames.append(df_a)

    if not fi_frames and not fa_frames:
        return None

    df_i = pd.concat(fi_frames).sort_index() if fi_frames else pd.DataFrame()
    df_a = pd.concat(fa_frames).sort_index() if fa_frames else pd.DataFrame()

    for df_ in [df_i, df_a]:
        if not df_.empty:
            df_ = df_[~df_.index.duplicated(keep="first")]

    # Merge instant + accum
    if df_i.empty:
        df = df_a.copy()
    elif df_a.empty:
        df = df_i.copy()
    else:
        df = df_i.join(df_a, how="outer", lsuffix="", rsuffix="_a")
        df.drop(columns=[c for c in df.columns if c.endswith("_a")],
                errors="ignore", inplace=True)

    df = df[~df.index.duplicated(keep="first")]
    df = apply_unit_conversions(df)
    df = compute_solar(df, lat, lon, alt)
    df = compute_sunrise_sunset(df, lat, lon, alt)
    df = finalize_df(df, lat, lon, alt, T_set, name,
                     cfg["dist"], cfg["cz"],
                     grid_lat=grid_lat_used, grid_lon=grid_lon_used)

    ghi_m = df["GHI"].mean() if "GHI" in df.columns else float("nan")
    t_m   = df["T_amb"].mean() if "T_amb" in df.columns else float("nan")
    ghi_s = f"{ghi_m:.1f} W/m²" if not np.isnan(ghi_m) else "NaN ⚠"
    t_s   = f"{t_m:.1f}°C"      if not np.isnan(t_m)   else "NaN ⚠"
    print(f"    ✓ {len(df):,} rows | GHI={ghi_s} | T_amb={t_s}")
    return df


# ═══════════════════════════════════════════════════════════
# PROCESS FULL GRID (all ~528 ERA5 points inside TN bbox)
# ═══════════════════════════════════════════════════════════

def process_full_grid(instant_ds, accum_ds):
    """
    Extract every ERA5 grid point inside the TN bounding box.
    Saves data/processed/grid/era5_TN_grid_all.csv
    Each row is one hour at one grid point.
    Columns include: timestamp, lat, lon, T_amb, RHum, GHI, DNI, DHI, ...
    """
    print("\n" + "─" * 56)
    print("  Processing full ERA5 grid (all ~528 grid points) ...")

    # Find grid coordinates from any available file
    sample_ds = None
    for year in YEARS:
        for month in MONTHS:
            if month in instant_ds.get(year, {}):
                sample_ds = instant_ds[year][month]
                break
        if sample_ds is not None:
            break

    if sample_ds is None:
        print("  [SKIP] No instant files found — cannot build grid CSV.")
        return

    lat_name = next((c for c in list(sample_ds.coords) + list(sample_ds.dims)
                     if "lat" in c.lower()), None)
    lon_name = next((c for c in list(sample_ds.coords) + list(sample_ds.dims)
                     if "lon" in c.lower()), None)

    if not lat_name or not lon_name:
        print("  [SKIP] Cannot find lat/lon dimensions in dataset.")
        return

    lats = sample_ds[lat_name].values.astype(float)
    lons = sample_ds[lon_name].values.astype(float)
    print(f"  Grid: {len(lats)} lat × {len(lons)} lon = {len(lats)*len(lons)} points")

    all_grid_frames = []

    for li, lat_val in enumerate(lats):
        for lo, lon_val in enumerate(lons):
            fi_frames, fa_frames = [], []

            for year in YEARS:
                for month in MONTHS:
                    if month in instant_ds.get(year, {}):
                        df_i = extract_grid_point(
                            instant_ds[year][month], li, lo, lat_val, lon_val)
                        if df_i is not None and len(df_i) > 0:
                            fi_frames.append(df_i)
                    if month in accum_ds.get(year, {}):
                        df_a = extract_grid_point(
                            accum_ds[year][month], li, lo, lat_val, lon_val)
                        if df_a is not None and len(df_a) > 0:
                            fa_frames.append(df_a)

            if not fi_frames and not fa_frames:
                continue

            df_i = pd.concat(fi_frames).sort_index() if fi_frames else pd.DataFrame()
            df_a = pd.concat(fa_frames).sort_index() if fa_frames else pd.DataFrame()

            for df_ in [df_i, df_a]:
                if not df_.empty:
                    df_ = df_[~df_.index.duplicated(keep="first")]

            if df_i.empty:
                df = df_a.copy()
            elif df_a.empty:
                df = df_i.copy()
            else:
                df = df_i.join(df_a, how="outer", lsuffix="", rsuffix="_a")
                df.drop(columns=[c for c in df.columns if c.endswith("_a")],
                        errors="ignore", inplace=True)

            df = df[~df.index.duplicated(keep="first")]
            df = apply_unit_conversions(df)
            df = compute_solar(df, lat_val, lon_val, alt=50)  # flat approx for grid
            # No sunrise/sunset for grid (too slow for 528 points × 365 days)

            # Time features
            df["timestamp"]   = df.index
            df["hour"]        = df.index.hour
            df["month"]       = df.index.month
            df["DOY"]         = df.index.dayofyear
            df["year"]        = df.index.year
            df["season"]      = df["month"].map(lambda m: SEASON_MAP[m][0])
            df["grid_lat"]    = lat_val
            df["grid_lon"]    = lon_val

            # Physical bounds
            if "GHI" in df.columns:
                df.loc[df["GHI"] < 0,    "GHI"] = 0
                df.loc[df["GHI"] > 1400, "GHI"] = np.nan
            if "T_amb" in df.columns:
                df.loc[df["T_amb"] < -5, "T_amb"] = np.nan
                df.loc[df["T_amb"] > 60, "T_amb"] = np.nan

            # Drop raw ERA5 cols
            keep_grid = {
                "timestamp", "grid_lat", "grid_lon",
                "T_amb", "T_dew", "RHum", "W_spd", "W_dir",
                "GHI", "DNI", "DHI", "avg_sdirswrf", "LW_down",
                "cloud_cover", "precipitation", "P_atm",
                "SZA", "solar_azimuth", "ETR", "GHI_clearsky", "CSI",
                "hour", "month", "DOY", "year", "season",
            }
            df.drop(columns=[c for c in df.columns if c not in keep_grid],
                    errors="ignore", inplace=True)
            df = df.reset_index(drop=True)
            all_grid_frames.append(df)

        pct = 100 * (li + 1) / len(lats)
        print(f"    lat row {li+1}/{len(lats)} ({lat_val:.2f}°N)  {pct:.0f}%", end="\r")

    print()

    if all_grid_frames:
        # ── Windows-safe CSV write: stream in chunks, never hold all rows ──
        # Avoids: OSError [Errno 22] (path too long or huge buffer on Windows)
        # Avoids: MemoryError (10M rows × N cols all in RAM at once)
        grid_path = os.path.join(OUTPUT_GRID, "era5_TN_grid_all.csv")

        # Use \\?\ prefix on Windows to bypass 260-char path limit
        if os.name == "nt":
            abs_path = os.path.abspath(grid_path)
            if not abs_path.startswith("\\\\?\\"):
                abs_path = "\\\\?\\" + abs_path
        else:
            abs_path = grid_path

        total_rows = 0
        header_written = False
        with open(abs_path, "w", newline="", encoding="utf-8") as f:
            for chunk_df in all_grid_frames:
                chunk_df.to_csv(f, index=False, header=not header_written)
                header_written = True
                total_rows += len(chunk_df)

        print(f"  ✓ Full grid CSV: {grid_path}")
        print(f"    Rows: {total_rows:,}  |  Grid points: {len(all_grid_frames)}")
        return grid_path
    else:
        print("  [WARN] No grid data produced.")
        return None


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Pre-flight check ──────────────────────────────────
    instant_count = sum(
        1 for y in YEARS for m in MONTHS
        if os.path.exists(
            os.path.join(INPUT_DIR, f"era5_TN_grid_{y}_{m}_instant.nc")))
    accum_count = sum(
        1 for y in YEARS for m in MONTHS
        if os.path.exists(
            os.path.join(INPUT_DIR, f"era5_TN_grid_{y}_{m}_accum.nc")))
    expected = len(YEARS) * len(MONTHS)

    print("=" * 68)
    print("  ERA5 COMBINE — TAMIL NADU  (All 260+ Locations + Full Grid)")
    print(f"  Input   : {INPUT_DIR}/")
    print(f"  Output  : {OUTPUT_NAMED}/  (per-location CSVs)")
    print(f"            {OUTPUT_GRID}/   (full grid CSV)")
    print(f"  Locations: {len(TN_LOCATIONS)}")
    print(f"  Files   : instant={instant_count}/{expected}"
          f"  accum={accum_count}/{expected}")

    if accum_count == 0:
        print("\n  ⚠️  No accum files found!")
        print("  Run 01_download_era5_tamilnadu.py first to download accum files.")
        print("  Without accum files, GHI / DNI / DHI / precipitation will be NaN.")
        ans = input("\n  Continue with instant-only data? (y/n): ").strip().lower()
        if ans != "y":
            print("  Exiting. Re-run after downloading accum files.")
            exit(1)

    if accum_count < expected:
        print(f"\n  ⚠️  Only {accum_count}/{expected} accum files present.")
        print("  Re-run 01_download_era5_tamilnadu.py to fill in the rest.")

    # ── Inspect first accum file for variable names ───────
    sample_fa = next(
        (os.path.join(INPUT_DIR, f"era5_TN_grid_{y}_{m}_accum.nc")
         for y in YEARS for m in MONTHS
         if os.path.exists(
             os.path.join(INPUT_DIR, f"era5_TN_grid_{y}_{m}_accum.nc"))),
        None)

    if sample_fa:
        ds_chk = open_nc(sample_fa)
        if ds_chk:
            avars = list(ds_chk.data_vars)
            print(f"\n  Accum variables found: {avars}")
            ds_chk.close()

    print("=" * 68)

    # ── Load all NetCDF files into memory ─────────────────
    instant_ds, accum_ds = load_all_files()

    # ════════════════════════════════════════════════════
    # PART 1: Named locations → individual CSVs
    # ════════════════════════════════════════════════════
    print(f"\n  Processing {len(TN_LOCATIONS)} named locations ...")
    print("─" * 68)

    processed, skipped = [], []

    for name, cfg in TN_LOCATIONS.items():
        print(f"\n  [{name}]  "
              f"lat={cfg['lat']:.4f}  lon={cfg['lon']:.4f}  "
              f"alt={cfg['alt']}m  dist={cfg['dist']}")
        df = process_location(name, cfg, instant_ds, accum_ds)
        if df is None:
            print(f"    [SKIP] No data found")
            skipped.append(name)
            continue
        out_path = os.path.join(OUTPUT_NAMED,
                                f"climate_{name.lower().replace(' ', '_')}.csv")
        abs_out = ("\\\\?\\" + os.path.abspath(out_path)
                   if os.name == "nt" else out_path)
        df.to_csv(abs_out, index=False)
        print(f"    → {out_path}")
        processed.append(name)
        # Free memory immediately — don't store 222 × 17k rows in RAM
        del df

    # Combined named-locations CSV — stream from already-saved per-location files
    # This avoids the MemoryError from pd.concat(222 DataFrames)
    cp = os.path.join(OUTPUT_COMBINED, "climate_tamilnadu_all.csv")
    abs_cp = ("\\\\?\\" + os.path.abspath(cp) if os.name == "nt" else cp)

    print("\n" + "─" * 68)
    print(f"  Building combined CSV from {len(processed)} location files ...")

    header_written = False
    total_rows = 0
    ghi_ok = 0

    with open(abs_cp, "w", newline="", encoding="utf-8") as fout:
        for name in processed:
            loc_path = os.path.join(OUTPUT_NAMED,
                                    f"climate_{name.lower().replace(' ', '_')}.csv")
            abs_loc = ("\\\\?\\" + os.path.abspath(loc_path)
                       if os.name == "nt" else loc_path)
            try:
                chunk = pd.read_csv(abs_loc)
                chunk.to_csv(fout, index=False, header=not header_written)
                header_written = True
                total_rows += len(chunk)
                if "GHI" in chunk.columns:
                    ghi_ok += chunk["GHI"].notna().sum()
                del chunk
            except Exception as e:
                print(f"    [WARN] Could not read {loc_path}: {e}")

    pct = 100 * ghi_ok / total_rows if total_rows else 0
    print(f"  Named locations combined: {cp}")
    print(f"  Rows: {total_rows:,}  |  Locations: {len(processed)}"
          f"  |  Skipped: {len(skipped)}")
    print(f"  GHI valid: {ghi_ok:,} / {total_rows:,}  ({pct:.0f}%)")

    # ════════════════════════════════════════════════════
    # PART 2: Full ERA5 grid → one big grid CSV
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 68)
    print("  Starting full grid extraction (~528 ERA5 points) ...")
    print("  This may take 30–90 minutes. Progress shown below.")
    print("=" * 68)
    grid_path = process_full_grid(instant_ds, accum_ds)

    # ── Close all open datasets ───────────────────────────
    for year in YEARS:
        for ds in instant_ds.get(year, {}).values():
            try: ds.close()
            except Exception: pass
        for ds in accum_ds.get(year, {}).values():
            try: ds.close()
            except Exception: pass

    # ── Final summary ─────────────────────────────────────
    print("\n" + "=" * 68)
    print("  ✅  ALL DONE")
    print()
    print(f"  Named CSVs  : {OUTPUT_NAMED}/climate_{{name}}.csv")
    print(f"                ({len(processed)} locations)")
    print(f"  Combined    : {OUTPUT_COMBINED}/climate_tamilnadu_all.csv")
    if grid_path:
        print(f"  Full grid   : {grid_path}")
    print("=" * 68)
    print("\nNext: load climate_tamilnadu_all.csv or era5_TN_grid_all.csv for training.")