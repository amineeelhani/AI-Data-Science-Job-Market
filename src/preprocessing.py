"""Preprocessing: data quality checks, cleaning, and a single table for plots and dashboard."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1] #resolve assoluto parents[1] sale di due livelli, file è dove sono ora
RAW = ROOT / "data" / "raw" / "AI Job Market Dataset.csv"
OUT = ROOT / "data" / "processed"

SKILL_COLS = ["skills_python", "skills_sql","skills_ml","skills_deep_learning","skills_cloud"]

df = pd.read_csv(RAW)
print("shape:", df.shape)
