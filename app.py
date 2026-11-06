from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Smart Cart | Customer Segmentation",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Smart Cart Customer Segmentation")
st.caption(
    "Explore the four customer groups produced by "
    "your notebook's Agglomerative Clustering model."
)

# Load the exported notebook results
csv_path = Path(__file__).parent / "clustered_customers.csv"

if not csv_path.is_file():
    st.error("Place clustered_customers.csv in the same folder as app.py.")
    st.stop()

# Read the small CSV on each rerun so new exports appear immediately
df = pd.read_csv(csv_path)

required_columns = [
    "Labels", "Income", "Total_Spending", "Age",
    "Total_Children", "PCA_1", "PCA_2", "PCA_3",
]

missing = [column for column in required_columns if column not in df.columns]

if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

if df.empty:
    st.warning("The exported dataset has no customers.")
    st.stop()

if df[required_columns].isna().any().any():
    st.error("Required columns contain missing values. Check your export.")
    st.stop()

# Treat cluster numbers as categories
df["Cluster"] = df["Labels"].map(lambda label: f"Cluster {label}")
cluster_order = sorted(df["Cluster"].unique())

colors = px.colors.qualitative.Safe
color_map = {
    cluster: colors[index % len(colors)]
    for index, cluster in enumerate(cluster_order)
}

# Sidebar filter
st.sidebar.header("Explore Customers")

selected_clusters = st.sidebar.multiselect(
    "Show clusters",
    options=cluster_order,
    default=cluster_order,
)

filtered_df = df[df["Cluster"].isin(selected_clusters)].copy()

st.sidebar.caption(
    "Cluster numbers are identifiers; a higher number "
    "does not mean a better customer."
)

if filtered_df.empty:
    st.info("Select at least one cluster in the sidebar.")
    st.stop()

# Metrics reflect the selected clusters
col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers Shown", f"{len(filtered_df):,}")
col2.metric("Clusters Shown", filtered_df["Cluster"].nunique())
col3.metric("Average Income", f"{filtered_df['Income'].mean():,.0f}")
col4.metric(
    "Average Total Spending",
    f"{filtered_df['Total_Spending'].mean():,.0f}",
)

overview_tab, plots_tab, data_tab = st.tabs(
    ["Cluster Overview", "Interactive Plots", "Customer Data"]
)

with overview_tab:
    st.subheader("Customers per Cluster")

    counts = (
        filtered_df.groupby("Cluster")
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        counts,
        x="Cluster",
        y="Customers",
        color="Cluster",
        text="Customers",
        color_discrete_map=color_map,
        category_orders={"Cluster": cluster_order},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Profiles")
    st.caption("Means calculated from the selected customers.")

    summary = (
        filtered_df.groupby("Cluster")
        .agg(
            Customers=("Cluster", "size"),
            Average_Income=("Income", "mean"),
            Average_Spending=("Total_Spending", "mean"),
            Average_Age=("Age", "mean"),
            Average_Children=("Total_Children", "mean"),
        )
        .round(1)
    )

    st.dataframe(summary, use_container_width=True)

with plots_tab:
    st.subheader("Income vs Total Spending")

    fig = px.scatter(
        filtered_df,
        x="Income",
        y="Total_Spending",
        color="Cluster",
        hover_data=["Age", "Total_Children"],
        color_discrete_map=color_map,
        category_orders={"Cluster": cluster_order},
        opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3D Cluster View")
    st.caption("Drag to rotate. Scroll to zoom.")

    fig_3d = px.scatter_3d(
        filtered_df,
        x="PCA_1",
        y="PCA_2",
        z="PCA_3",
        color="Cluster",
        hover_data=["Income", "Total_Spending", "Age"],
        color_discrete_map=color_map,
        category_orders={"Cluster": cluster_order},
        opacity=0.7,
    )

    fig_3d.update_traces(marker={"size": 3})
    fig_3d.update_layout(height=650)

    st.plotly_chart(fig_3d, use_container_width=True)

    st.caption(
        "These PCA axes combine the original features. "
        "Your notebook's three components retain about 45% "
        "of the total variance."
    )

with data_tab:
    st.subheader("Selected Customers")

    st.dataframe(
        filtered_df.drop(columns=["Labels"]),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download Selected Customers",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="selected_customers.csv",
        mime="text/csv",
    )