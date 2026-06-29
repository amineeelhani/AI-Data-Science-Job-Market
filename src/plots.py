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

#Figure 2: Role market share over time
t = (df.groupby(["job_posting_year", "job_title"], observed=True)["job_openings"]
       .sum().reset_index(name="openings"))
t["share"] = t["openings"] / t.groupby("job_posting_year")["openings"].transform("sum") * 100 #the same value for every row (year)
pivot = t.pivot(index="job_posting_year", columns="job_title", values="share")
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for role, color in ROLE_COLORS.items():
    ax.plot(pivot.index, pivot[role], "o-", label=role, color=color,
            linewidth=1.4, markersize=3)
ax.set_title("Share of advertised openings by job title, 2020-2026")
ax.set_ylabel("% of yearly openings")
ax.set_ylim(0, 25)
ax.legend( fontsize=7.5, ncol=2)
fig.savefig(FIGDIR / "fig02_temporal_shares.png") 
plt.close(fig)

#Figure 3: Salary distribution and boxplot by role
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].hist(df["salary"] / 1000, bins=40, color=NEUTRAL)
axes[0].set_title("Salary distribution (all postings)")
axes[0].set_xlabel("Annual salary (k USD)")
axes[0].set_ylabel("Postings")
order = df.groupby("job_title", observed=True)["salary"].median().sort_values(ascending=False).index
sns.boxplot(data=df.assign(salary_k=df["salary"] / 1000),
            x="salary_k", y="job_title", order=order, hue="job_title",
            palette=ROLE_COLORS, legend=False, fliersize=1, linewidth=0.8, ax=axes[1])
axes[1].set_title("Salary by job title")
axes[1].set_xlabel("Annual salary (k USD)")
axes[1].set_ylabel("")
fig.savefig(FIGDIR / "fig03_salary_distribution.png") 
plt.close(fig)

# Figure 4: Signal vs Noise
drivers = {
    "Job title": "job_title", "Experience level": "experience_level",
    "Company size": "company_size", "Hiring urgency": "hiring_urgency",
    "Country": "country", "Industry": "company_industry",
    "Education level": "education_level", "Remote type": "remote_type",
}
rows = [{"driver": label,
         "spread": (df.groupby(col, observed=True)["salary"].mean().max()
                    - df.groupby(col, observed=True)["salary"].mean().min()) / 1000}
        for label, col in drivers.items()]
spread = pd.DataFrame(rows).sort_values("spread")

fig, ax = plt.subplots(figsize=(7, 3.4))
colors = [ACCENT if s > 5 else NEUTRAL for s in spread["spread"]]
ax.barh(spread["driver"], spread["spread"], color=colors)
ax.bar_label(ax.containers[0], fmt="{:,.1f}", fontsize=8, padding=3)
ax.set_xlim(0, spread["spread"].max()*1.12)
ax.set_title("Salary spread across the levels of each variable (k USD)")
ax.set_xlabel("max(group mean) − min(group mean), k USD")
fig.savefig(FIGDIR / "fig04_salary_drivers.png")
plt.close(fig)

#Figure 5: Skill vs Seniority Heatmap
levels = ["Entry", "Mid", "Senior"]
matrix = pd.DataFrame(index=list(SKILL_COLS.values()), columns=levels, dtype=float)
for col, label in SKILL_COLS.items():
    for lvl in levels:
        sub = df[(df[col] == 1) & (df["experience_level"] == lvl)]
        matrix.loc[label, lvl] = wmean(sub) / 1000

fig, ax = plt.subplots(figsize=(6, 3.6))
sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlOrBr",
            cbar_kws={"label": "Mean salary (k USD)"}, linewidths=0.5,
            linecolor="white", ax=ax)
ax.set_title("Mean salary by skill and seniority (weighted)")
fig.savefig(FIGDIR / "fig05_skill_seniority.png")
plt.close(fig)

#Figure 6: Skill salary premium
lift = []
for col, label in SKILL_COLS.items():
    have = wmean(df[df[col] == 1])
    miss = wmean(df[df[col] == 0])
    lift.append({"skill": label, "lift": (have - miss) / 1000})
lift = pd.DataFrame(lift).sort_values("lift")

fig, ax = plt.subplots(figsize=(6.5, 3.2)) 
colors = [ACCENT if v > 1 else NEUTRAL for v in lift["lift"]]
ax.barh(lift["skill"], lift["lift"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.bar_label(ax.containers[0], fmt="{:+,.1f}", fontsize=8, padding=3)
ax.set_xlim(lift["lift"].min() - 3, lift["lift"].max() * 1.15)
ax.set_title("Salary premium for having each skill (weighted)")
ax.set_xlabel("Mean salary difference: has skill − lacks skill (k USD)")
fig.savefig(FIGDIR / "fig06_skill_premium.png")
plt.close(fig)

#Figure 7: Geography
geo = (df.groupby("country", observed=True)
         .apply(lambda g: pd.Series({
             "openings": g["job_openings"].sum(),
             "salary_k": wmean(g) / 1000}))
         .sort_values("salary_k"))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
axes[0].barh(geo.index, geo["openings"], color=NEUTRAL)
axes[0].set_title("Advertised openings by country (weighted)")
axes[0].set_xlabel("Openings")

axes[1].barh(geo.index, geo["salary_k"], color=NEUTRAL)
axes[1].set_title("Mean salary by country (weighted)")
axes[1].set_xlabel("Mean salary (k USD)")
axes[1].bar_label(axes[1].containers[0], fmt="{:,.1f}", fontsize=8, padding=3)
axes[1].set_xlim(0, 135)
fig.savefig(FIGDIR / "fig07_geography.png")
plt.close(fig)

#Figure 8: Internal Inconsistencies
rng = np.random.default_rng(0)
sample = df.sample(2500, random_state=0)
jitter = rng.uniform(-0.25, 0.25, len(sample))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].scatter(sample["years_experience"] + jitter, sample["salary"] / 1000,
                s=4, alpha=0.25, color=NEUTRAL, edgecolors="none")
r = df["salary"].corr(df["years_experience"])
axes[0].set_title(f"Years of experience vs salary (r = {r:.3f})")
axes[0].set_xlabel("Years of experience")
axes[0].set_ylabel("Salary (k USD)")

sns.boxplot(data=df, x="experience_level", y="years_experience",
            order=["Entry", "Mid", "Senior"], color=NEUTRAL,
            linewidth=0.8, fliersize=1, ax=axes[1])
axes[1].set_title("Years of experience by declared seniority")
axes[1].set_xlabel("")
axes[1].set_ylabel("Years of experience")
axes[1].annotate("identical distributions:\ncolumns generated independently",
                 xy=(0.30, 0.85), xycoords="axes fraction", fontsize=9, color=ACCENT)
fig.savefig(FIGDIR / "fig08_inconsistencies.png")
plt.close(fig)