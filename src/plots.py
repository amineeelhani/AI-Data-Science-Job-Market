from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "report" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

ROLE_COLORS = {
    "AI Engineer": "#D55E00",
    "Machine Learning Engineer": "#E69F00",
    "Data Engineer": "#0072B2",
    "Data Scientist": "#009E73",
    "Data Analyst": "#56B4E9",
    "Business Analyst": "#CC79A7",
}

ACCENT = "#D55E00"    # orange: signal
NEUTRAL = "#7F7F7F"   # grey: context/noise


sns.set_theme(style="white", context="paper", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

df = pd.read_parquet(ROOT / "data" / "processed" / "jobs_clean.parquet")
SKILL_COLS = {
    "skills_python": "Python", "skills_sql": "SQL", "skills_ml": "Machine Learning",
    "skills_deep_learning": "Deep Learning", "skills_cloud": "Cloud",
}

def wmean(g, col="salary"):
    return (g[col] * g["job_openings"]).sum() / g["job_openings"].sum()

#Figure 1: Volume per year
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
yearly = df.groupby("job_posting_year").agg(
    postings=("job_id", "size"), openings=("job_openings", "sum"))
for ax, col, title in [(axes[0], "postings", "Posting rows per year"),
                       (axes[1], "openings", "Advertised openings per year (weighted)")]:
    ax.bar(yearly.index, yearly[col], color=NEUTRAL, width=0.7)
    ax.set_title(title)
    ax.set_ylim(0, yearly[col].max() * 1.15)
    ax.bar_label(ax.containers[0], fmt="{:,.0f}", fontsize=8, padding=2)
    ax.set_yticks([])
fig.savefig(FIGDIR / "fig01_volume_per_year.png") 
plt.close(fig)