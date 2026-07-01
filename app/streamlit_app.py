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
st.sidebar.caption(" Synthetic dataset: generated using statistical distributions and controlled randomness")

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

#TAB 1: Market evolution
with tab1:
    st.subheader("How the role mix evolves over time")
    st.info("How to read it: shares stay remarkably flat: no role overtakes another. "
            "The market stays flat in both volume and composition.")
    
    mode = st.radio("Show as", ["Absolute openings", "Share (%)"],
                    horizontal=True, key="evo_mode")

    ev = (fdf.groupby(["job_posting_year", "job_title"], observed=True)["job_openings"]
            .sum().reset_index(name="openings"))

    if ev["job_posting_year"].nunique() < 2:
        st.info("Select a range of at least two years to see the evolution over time.")
    else:
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



#TAB 2: Skills & salary
with tab2:
    st.subheader("Where the salary signal really is: skills and seniority")
    st.info("How to read it: seniority is the strongest driver, advanced skills "
            "(ML, Deep Learning, Cloud) add a real premium, while Python and SQL are "
            "now baseline and barely move pay.")
    skill_cols = {"skills_python": "Python", "skills_sql": "SQL", "skills_ml": "Machine Learning",
                  "skills_deep_learning": "Deep Learning", "skills_cloud": "Cloud"}
    levels = ["Entry", "Mid", "Senior"]

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("**Mean salary by skill and seniority (k USD, weighted)**")
        rows = []
        for col, label in skill_cols.items():
            for lvl in levels:
                sub = fdf[(fdf[col] == 1) & (fdf["experience_level"] == lvl)]
                rows.append({"skill": label, "level": lvl,
                             "salary_k": round(wmean(sub) / 1000, 1) if len(sub) else None})
        heat = pd.DataFrame(rows).pivot(index="skill", columns="level", values="salary_k")
        heat = heat[levels]
        fig_h = px.imshow(heat, text_auto=True, aspect="auto",
                          color_continuous_scale="YlOrBr",
                          labels={"color": "k USD"})
        fig_h.update_layout(height=380)
        st.plotly_chart(fig_h, use_container_width=True)
    
    with col_right:
        st.markdown("**Salary premium per skill (k USD, weighted)**")
        lift_rows = []
        for col, label in skill_cols.items():
            have = fdf[fdf[col] == 1]
            miss = fdf[fdf[col] == 0]
            if len(have) and len(miss):
                lift_rows.append({"skill": label, "lift": (wmean(have) - wmean(miss)) / 1000})
        lift = pd.DataFrame(lift_rows).sort_values("lift")
        fig_l = px.bar(lift, x="lift", y="skill", orientation="h",
                       labels={"lift": "has - lacks (k USD)", "skill": ""})
        fig_l.update_traces(marker_color=["#7F7F7F" if v < 1 else "#D55E00" for v in lift["lift"]])
        fig_l.update_layout(height=380)
        st.plotly_chart(fig_l, use_container_width=True)

#TAB 3: Global landscape
with tab3:
    st.subheader("A geographically flat market")
    st.info("How to read it: salaries and volumes are nearly identical across countries "
            "(spread ≈ 3k USD). The map shows where postings are, not meaningful pay "
            "differences: read it together with the table.")

    geo = (fdf.groupby("country", observed=True)
              .apply(lambda g: pd.Series({
                  "openings": int(g["job_openings"].sum()),
                  "salary_k": round(wmean(g) / 1000, 1)}), include_groups=False)
              .reset_index())

    col_map, col_tab = st.columns([1.6, 1])

    with col_map:
        fig_geo = px.scatter_geo(
            geo, locations="country", locationmode="country names",
            size="openings", color="salary_k",
            color_continuous_scale="YlOrBr", range_color = [45,205], projection="natural earth",
            labels={"salary_k": "Mean salary (k USD)", "openings": "Openings"})
        fig_geo.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_geo, use_container_width=True)

    with col_tab:
        st.markdown("**Exact values**")
        table = geo.sort_values("salary_k", ascending=False).rename(
            columns={"country": "Country", "openings": "Openings", "salary_k": "Salary (k USD)"})
        st.dataframe(table, hide_index=True, use_container_width=True)

    st.caption(f"Salary range across countries: "
               f"{geo['salary_k'].min():.1f} - {geo['salary_k'].max():.1f} k USD "
               f"(spread {geo['salary_k'].max() - geo['salary_k'].min():.1f} k).")

#TAB 4: Success formula
with tab4:
    st.subheader("What actually predicts pay (and what doesn't)")
    st.info("How to read it: pay is driven by seniority and company size. Education is a "
            "null result: a PhD pays no more than a Bachelor. Years of experience is "
            "excluded: in this dataset it is inconsistent with declared seniority.")


    st.markdown("**Mean salary by seniority and company size (k USD, weighted)**")
    sizes = ["Startup", "Medium", "Enterprise", "MNC"]
    levels = ["Entry", "Mid", "Senior"]
    grp = (fdf.groupby(["experience_level", "company_size"], observed=True)
              .apply(lambda g: round(wmean(g) / 1000, 1), include_groups=False)
              .reset_index(name="salary_k"))
    fig_b = px.bar(grp, x="experience_level", y="salary_k", color="company_size",
                   barmode="group", category_orders={"experience_level": levels,
                   "company_size": sizes},
                   labels={"experience_level": "", "salary_k": "Mean salary (k USD)",
                           "company_size": "Company size"})
    fig_b.update_layout(height=420)
    st.plotly_chart(fig_b, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Education level: null result**")
        edu = (fdf.groupby("education_level", observed=True)
                  .apply(lambda g: round(wmean(g) / 1000, 1), include_groups=False)
                  .reset_index(name="salary_k"))
        fig_e = px.bar(edu, x="education_level", y="salary_k",
                       category_orders={"education_level": ["Bachelor", "Master", "PhD"]},
                       labels={"education_level": "", "salary_k": "Mean salary (k USD)"})
        fig_e.update_traces(marker_color="#7F7F7F")
        fig_e.update_layout(height=340)
        st.plotly_chart(fig_e, use_container_width=True)
    with col_b:
        st.markdown("**The proposal's original idea**")
        st.caption("We initially planned a years-of-experience vs salary scatter. "
                   "We kept it here as documentation: the relationship is flat (r ≈ 0), " 
                   "which is why it did not become a main view.")
        with st.expander("Show the abandoned scatter"):
            scatter_df = fdf.sample(min(2000, len(fdf)), random_state=0)
            fig_s = px.scatter(scatter_df, x="years_experience", y="salary",
                               opacity=0.25, labels={"years_experience": "Years of experience",
                                                     "salary": "Salary (USD)"})
            fig_s.update_traces(marker_color="#7F7F7F", marker_size=4)
            fig_s.update_layout(height=340)
            st.plotly_chart(fig_s, use_container_width=True)

