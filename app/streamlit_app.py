from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ROLE_COLORS = {
    "AI Engineer": "#D55E00",
    "Machine Learning Engineer": "#E69F00",
    "Data Engineer": "#0072B2",
    "Data Scientist": "#009E73",
    "Data Analyst": "#56B4E9",
    "Business Analyst": "#CC79A7",
}

st.set_page_config(page_title="AI & DS Job Market", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet(ROOT / "data" / "processed" / "jobs_clean.parquet")


df = load_data()

def wmean(g, col="salary"):
    return (g[col] * g["job_openings"]).sum() / g["job_openings"].sum()

#Sidebar, Global Filters
st.sidebar.title("Filters")
st.sidebar.caption(" Synthetic dataset - generated using statistical distributions and controlled randomness")

years = sorted(df["job_posting_year"].unique())
year_range = st.sidebar.slider(
    "Posting year", min_value=int(years[0]), max_value=int(years[-1]),
    value=(int(years[0]), int(years[-1])))

countries = sorted(df["country"].unique())
sel_countries = st.sidebar.multiselect("Countries", countries, default=countries)

roles = list(ROLE_COLORS.keys())
sel_roles = st.sidebar.multiselect("Job titles", roles, default=roles)

mask = (
    df["job_posting_year"].between(*year_range)
    & df["country"].isin(sel_countries)
    & df["job_title"].isin(sel_roles)
)
fdf = df[mask]

st.title("The AI & Data Science Job Market (2020–2026)")
st.caption(f"{len(fdf):,} postings selected - {fdf['job_openings'].sum():,} weighted openings")

if fdf.empty:
    st.warning("No data matches the current filters. Widen your selection.")
    st.stop()


tab1, tab2, tab3, tab4 = st.tabs(
    ["Market evolution", "Skills & salary",
     "Global landscape", "Success formula"]
)

#TAB 1
with tab1:
    st.subheader("How the role mix evolves over time")
    st.info("How to read it: shares stay remarkably flat: no role overtakes another. "
            "The market grows in volume, not in composition.")
    
    mode = st.radio("Show as", ["Absolute openings", "Share (%)"],
                    horizontal=True, key="evo_mode")

    ev = (fdf.groupby(["job_posting_year", "job_title"], observed=True)["job_openings"]
            .sum().reset_index(name="openings"))

    if mode == "Share (%)":
        ev["value"] = (ev["openings"]
                       / ev.groupby("job_posting_year")["openings"].transform("sum") * 100)
        ylabel = "% of yearly openings"
    else:
        ev["value"] = ev["openings"]
        ylabel = "Advertised openings"

    fig = px.area(
        ev, x="job_posting_year", y="value", color="job_title",
        color_discrete_map=ROLE_COLORS, labels={"value": ylabel, "job_posting_year": "Year"})
    fig.update_layout(legend_title_text="", height=460)
    st.plotly_chart(fig, use_container_width=True)


