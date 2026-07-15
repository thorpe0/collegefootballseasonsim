"""Project-wide configuration and paths."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CFBD_API_KEY = os.environ.get("CFBD_API_KEY")

# Seasons to pull. 2021 onward = post-COVID, normal-length schedules.
# Bump END_SEASON once a new season finishes.
START_SEASON = 2021
END_SEASON = 2025

SEASONS = list(range(START_SEASON, END_SEASON + 1))
