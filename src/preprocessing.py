"""Preprocessing: data quality checks, cleaning, and a single table for plots and dashboard."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1] #resolve assoluto parents[1] sale di due livelli, file è dove sono ora
RAW = ROOT / "data" / "raw" / "AI Job Market Dataset.csv"
OUT = ROOT / "data" / "processed"

SKILL_COLS = ["skills_python", "skills_sql","skills_ml","skills_deep_learning","skills_cloud"]

df = pd.read_csv(RAW)
print("shape:", df.shape)
pd.set_option("display.max_columns", None)
df.info()
print(df.head())
print(df.describe())

print("missing values total:",df.isna().sum().sum())
print("duplicated rows:", df.duplicated().sum())
assert df.isna().sum().sum() == 0
assert df.duplicated().sum() == 0
assert df["job_id"].is_unique

assert df["salary"].between(40_000, 250_000).all()
assert df["job_posting_year"].between(2020, 2026).all()
assert df["job_posting_month"].between(1, 12).all()
assert df["job_openings"].between(1, 9).all()
assert df[SKILL_COLS].isin([0, 1]).all().all()

#internal consistency: experience_level vs years_experience
# key inconsistency of this dataset: the means are circa 7 years for all levels
# the two columns were generated independently of each other.
print(df.groupby("experience_level")["years_experience"].mean())
print("corr(salary, years_experience):",
      df["salary"].corr(df["years_experience"]).round(3))

ORDERS = {
    "experience_level" : ["Entry", "Mid", "Senior"],
    "education_level" : ["Bachelor","Master","PhD"],
    "company_size" : ["Startup","Medium", "Enterprise","MNC"],
    "hiring_urgency" : ["Low", "Medium", "High"],
}

for col, order in ORDERS.items():
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)

df["posting_date"] = pd.to_datetime(
    dict(year=df["job_posting_year"], month=df["job_posting_month"], day=1)
)

df["n_skills"] = df[SKILL_COLS].sum(axis=1)

def wmean(g: pd.DataFrame, col: str = "salary") -> float:
    """Weighted mean by job_openings: each row counts as 1-9 positions"""
    return (g[col] * g["job_openings"]).sum() / g["job_openings"].sum()