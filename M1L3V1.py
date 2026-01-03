# import packages
import streamlit as st
import pandas as pd
import re
import os
from pathlib import Path
import plotly.express as px


# Helper function to get dataset path
def get_dataset_path():
    base_dir = Path(__file__).resolve().parent
    return base_dir / "data" / "customer_reviews" / "customer_reviews.csv"


# Helper function to clean text
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


st.title("Hello, GenAI!")
st.write("This is your GenAI-powered data processing app.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            if not csv_path.exists():
                st.error(f"Dataset not found at: {csv_path}")
            else:
                df = pd.read_csv(csv_path)

                # Normalize column names to uppercase for simpler matching
                df.rename(columns={c: c.upper() for c in df.columns}, inplace=True)

                # Create expected columns when possible
                if "SUMMARY" not in df.columns and "REVIEW_TEXT" in df.columns:
                    df["SUMMARY"] = df["REVIEW_TEXT"]
                if "PRODUCT" not in df.columns:
                    df["PRODUCT"] = "Unknown Product"

                required_cols = {"SUMMARY", "PRODUCT"}
                missing = required_cols - set(df.columns)
                if missing:
                    st.error(f"Dataset is missing columns: {', '.join(sorted(missing))}")
                else:
                    st.session_state["df"] = df
                    st.success("Dataset loaded successfully!")
                    st.caption(f"Loaded from: {csv_path}")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🧹 Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("Reviews parsed and cleaned!")
        else:
            st.warning("Please ingest the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)


    st.subheader("Sentiment Score by Product")
    grouped = st.session_state["df"].groupby("PRODUCT")["SENTIMENT_SCORE"].mean()
    st.bar_chart(grouped)

    st.subheader("Distribution of Sentiment Scores")
    fig = px.histogram(
        filtered_df,
        x="SENTIMENT_SCORE",
        nbins=10,
        title="Distribution of Sentiment Scores",
        labels={
            "SENTIMENT_SCORE": "Sentiment Score",
            "count": "Frequency"
        }
    )
    fig.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Frequency",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
