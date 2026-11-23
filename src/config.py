from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "student_data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_REGIONS = [
    "Asia Pacific",
    "Europe",
    "India",
    "Latin America",
    "Middle East & Africa",
    "North America",
]

CANONICAL_CHANNELS = [
    "Aftermarket",
    "Direct OEM",
    "Export",
    "Tier-1 Distributor",
]

CANONICAL_FAMILIES = [
    "Brake & Safety Systems",
    "Chassis & Structural",
    "Electrical Assemblies",
    "Hydraulic Systems",
    "Powertrain Components",
    "Precision Machined Parts",
    "Thermal Management",
]

TRAIN_START = "2019-01"
TRAIN_END = "2026-10"
TEST_START = "2026-11"
TEST_END = "2028-05"
FORECAST_HORIZON = 19
