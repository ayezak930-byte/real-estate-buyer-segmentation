
import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------
# Page Configuration
# -------------------------------------------
st.set_page_config(
    page_title="Real Esate Buyer Intelligence",
    page_icon="🏠",
    layout="wide"
)

# ------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("clients_clustered.csv")

client = load_data()

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("\U0001F3E0 Real Estate Buyer Segmentation & Investment Intelligence")
st.markdown(
    "Machine Learning based Buyer Segmentation for Real Estate Intelligence"
)

# ------------------------------------------------------------------------------
# Sidebar Filters
# ------------------------------------------------------------------------------
st.sidebar.header(" filters")

country_options = ["ALL"] + sorted(client["country"].dropna().unique().tolist())
region_options = ["ALL"] + sorted(client["region"].dropna().unique().tolist())
purpose_options = ["ALL"] + sorted(client["acquisition_purpose"].dropna().unique().tolist())
type_options = ["ALL"] + sorted(client["client_type"].dropna().unique().tolist())

selected_country = st.sidebar.selectbox("Country", country_options)
selected_region = st.sidebar.selectbox("Region", region_options)
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", purpose_options)
selected_type = st.sidebar.selectbox("Client Type", type_options)

filtered_data = client.copy()

if selected_country != "ALL":
  filtered_data = filtered_data[filtered_data["country"] == selected_country]
if selected_region != "ALL":
  filtered_data = filtered_data[filtered_data["region"] == selected_region]
if selected_purpose != "ALL":
  filtered_data = filtered_data[filtered_data["acquisition_purpose"] == selected_purpose]
if selected_type != "ALL":
  filtered_data = filtered_data[filtered_data["client_type"] == selected_type]

# -----------------------------------------------------------------------------
# KPI Cards
# -----------------------------------------------------------------------------
st.subheader(" Buyer Segmentation Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Buyers", len(filtered_data))
col2.metric("Clusters", filtered_data["cluster"].nunique())
col3.metric(
    "Investment Buyers", filtered_data["acquisition_purpose"].value_counts().get("Investment", 0)
)
col4.metric(
   "Loan Applicants", (filtered_data["loan_applied"] == "Yes").sum()
)

# ---------------------------------------------------------------
# Buyer Segmentation
# ---------------------------------------------------------------
st.subheader("Buyer Distribution by Cluster")

cluster_counts = filtered_data["cluster"].value_counts().sort_index().reset_index()
cluster_counts.columns = ["Cluster", "Buyers"]

fig_cluster = px.bar(
    cluster_counts,
    x="Cluster",
    y="Buyers",
    title="Number of Buyers by Clusters",
    text="Buyers"
)

st.plotly_chart(fig_cluster, use_container_width=True)

# ------------------------------------------------------
# Investor Behavior Dashboard
# ------------------------------------------------------
st.subheader(" Investor Behavior")

purpose_cluster = pd.crosstab(
    filtered_data["cluster"],
    filtered_data["acquisition_purpose"]
).reset_index()

purpose_long = purpose_cluster.melt(
    id_vars="cluster",
    var_name="Acquisition Purpose",
    value_name="Number of Buyers"
)

fig_purpose = px.bar(
    purpose_long,
    x="cluster",
    y="Number of Buyers",
    color="Acquisition Purpose",
    barmode="group",
    title="Acquisition Purpose by Cluster"
)

st.plotly_chart(fig_purpose, use_container_width=True)

# -------------------------------------------------------
# Loan Behavior
# -------------------------------------------------------
loan_cluster = pd.crosstab(
    filtered_data["cluster"],
    filtered_data["loan_applied"]
).reset_index()

loan_long = loan_cluster.melt(
    id_vars="cluster",
    var_name="Loan Applied",
    value_name="Number of Buyers"
)

fig_loan = px.bar(
    loan_long,
    x="cluster",
    y="Number of Buyers",
    color="Loan Applied",
    barmode="group",
    title="Loan Behavior by Cluster"
)

st.plotly_chart(fig_loan, use_container_width=True)

# ------------------------------------------------------
# Geographic Buyer Analysis
# ------------------------------------------------------
st.subheader("Geographic Buyer Analysis")

country_cluster = (
    filtered_data.groupby(["country", "cluster"])
    .size()
    .reset_index(name="Buyers")
)

fig_geo = px.bar(
    country_cluster,
    x="country",
    y="Buyers",
    color="cluster",
    barmode="group",
    title="Buyers by Country and Cluster"
)

st.plotly_chart(fig_geo, use_container_width=True)

# ----------------------------------------------------
# Segment Insight Panel
# ----------------------------------------------------
st.subheader("Segment Insight")

segment_summary = (
    filtered_data.groupby("cluster")
    .agg(
        Buyers=("cluster", "count"),
        Average_Age=("age", "mean"),
        Average_Satisfaction=("satisfaction_score", "mean"),
        Investment_Buyers=("acquisition_purpose",
                           lambda x: (x == "Investment").sum()),
        Loan_Applicant=("loan_applied",
                        lambda x: (x == "Yes").sum())
    )
    .reset_index()
)

segment_summary["Investment_Percentage"] = (segment_summary["Investment_Buyers"] / segment_summary["Buyers"] * 100).round(1)
segment_summary["Average_Satisfaction"] = segment_summary[
    "Average_Satisfaction"
    ].round(2)

st.dataframe(
    segment_summary,
    use_container_width=True,
    hide_index=True
)

# ----------------------------------------------
# Cluster Profile
# ----------------------------------------------
st.subheader("Cluster Profiles")

cluster_ptofiles = {
    0: "Small buyer segment with comparatively low population.",
    1: "Large buyer segment with substantial loan and purchase activity.",
    2: "Large buyer segment with substantial loan and purchase activity.",
    3: "Smaller buyer segment with distinct buyer characteristics."
}

for cluster in sorted(filtered_data["cluster"].unique()):
    st.markdown(
        f"**Cluster {cluster}**"
        f"{cluster_ptofiles.get(cluster, 'Buyer segment identified through clustering.')}"
    )

# -----------------------------------------------------------------------------------
# Data Preview
# ------------------------------------------------------------------------------------
with st.expander("View Filtered Buyer Data"):
  st.dataframe(filtered_data, use_container_width=True)
