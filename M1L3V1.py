# import packages
import streamlit as st
import pandas as pd
import re
import os
from pathlib import Path
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv(".env.local")


# Helper function to get dataset path
def get_dataset_path():
    base_dir = Path(__file__).resolve().parent
    return base_dir / "data" / "customer_reviews" / "customer_reviews.csv"


# Helper function to clean text
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


# Categorize sentiment based on score
def categorize_sentiment(score):
    if score > 0.3:
        return "Positive"
    elif score < -0.3:
        return "Negative"
    else:
        return "Neutral"


# Get marketing insights using OpenAI
def get_marketing_insights(df, api_key):
    client = OpenAI(api_key=api_key)
    
    # Get positive reviews
    positive_reviews = df[df["SENTIMENT_SCORE"] > 0.3]["SUMMARY"].head(10).tolist()
    
    prompt = f"""Analyze these positive customer reviews and provide:
1. Top 3 key themes customers love
2. Which products to prioritize for marketing

Reviews:
{chr(10).join(positive_reviews)}

Keep the response concise and actionable."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    usage = response.usage
    return response.choices[0].message.content, usage


# Extract key themes from reviews
def extract_themes(reviews, api_key):
    client = OpenAI(api_key=api_key)
    
    sample_reviews = reviews[:5]
    
    prompt = f"""Extract the top 5 most mentioned features or themes from these reviews.
Return as a simple comma-separated list.

Reviews:
{chr(10).join(sample_reviews)}"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    
    usage = response.usage
    return response.choices[0].message.content, usage


# Analyze negative reviews for improvements
def analyze_negative_reviews(df, api_key):
    client = OpenAI(api_key=api_key)
    
    # Get negative reviews
    negative_reviews = df[df["SENTIMENT_SCORE"] < -0.3]["SUMMARY"].head(15).tolist()
    
    if not negative_reviews:
        return "No significant negative reviews found.", None
    
    prompt = f"""Analyze these negative customer reviews and provide:
1. Top 3 critical issues customers complain about

2. Specific actionable fixes to address complaints

Reviews:
{chr(10).join(negative_reviews)}

Be direct and actionable. Prioritize by severity."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    usage = response.usage
    return response.choices[0].message.content, usage


# Identify top-rated products customers love
def identify_top_products(df, api_key):
    client = OpenAI(api_key=api_key)
    
    # Get product sentiment averages
    product_sentiment = df.groupby("PRODUCT")["SENTIMENT_SCORE"].agg(['mean', 'count']).sort_values('mean', ascending=False)
    
    # Get top products with positive sentiment
    top_products = product_sentiment[product_sentiment['mean'] > 0.2].head(5)
    
    if top_products.empty:
        return "No products with significantly positive sentiment found.", None
    
    # Get sample positive reviews for top products
    top_product_names = top_products.index.tolist()
    sample_reviews = []
    for product in top_product_names[:3]:
        reviews = df[(df["PRODUCT"] == product) & (df["SENTIMENT_SCORE"] > 0.3)]["SUMMARY"].head(3).tolist()
        sample_reviews.extend(reviews)
    
    prompt = f"""Based on these highly-rated product reviews, identify:
1. Which specific products customers love the most and why
2. Key features that make these products stand out
3. Marketing angles to emphasize for these winning products

Top products by sentiment: {', '.join(top_product_names)}

Sample reviews:
{chr(10).join(sample_reviews)}

Focus on actionable insights for promoting these successful products."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    usage = response.usage
    return response.choices[0].message.content, usage


st.title("Hello, GenAI!")
st.write("This is your GenAI-powered data processing app.")

# Load API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Initialize usage tracking in session state
if "total_tokens" not in st.session_state:
    st.session_state["total_tokens"] = 0
    st.session_state["total_cost"] = 0.0

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
            st.session_state["df"]["SENTIMENT_CATEGORY"] = st.session_state["df"]["SENTIMENT_SCORE"].apply(categorize_sentiment)
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

    # AI-Powered Marketing Insights
    if api_key:
        st.subheader("🤖 AI Marketing Insights")
        
        col_ai1, col_ai2, col_ai3, col_ai4 = st.columns(4)
        
        with col_ai1:
            if st.button("📊 Get Marketing Strategy"):
                with st.spinner("Analyzing reviews with AI..."):
                    try:
                        insights, usage = get_marketing_insights(st.session_state["df"], api_key)
                        st.markdown(insights)
                        if usage:
                            st.session_state["total_tokens"] += usage.total_tokens
                            # GPT-4 pricing: $0.03 per 1K prompt tokens, $0.06 per 1K completion tokens
                            cost = (usage.prompt_tokens / 1000 * 0.03) + (usage.completion_tokens / 1000 * 0.06)
                            st.session_state["total_cost"] += cost
                            st.caption(f"Tokens used: {usage.total_tokens} (~${cost:.4f})")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col_ai2:
            if st.button("🔑 Extract Key Themes"):
                with st.spinner("Extracting themes with AI..."):
                    try:
                        positive_reviews = filtered_df[filtered_df["SENTIMENT_SCORE"] > 0.3]["SUMMARY"].tolist()
                        if positive_reviews:
                            themes, usage = extract_themes(positive_reviews, api_key)
                            st.markdown(f"**Key Themes:** {themes}")
                            if usage:
                                st.session_state["total_tokens"] += usage.total_tokens
                                cost = (usage.prompt_tokens / 1000 * 0.03) + (usage.completion_tokens / 1000 * 0.06)
                                st.session_state["total_cost"] += cost
                                st.caption(f"Tokens used: {usage.total_tokens} (~${cost:.4f})")
                        else:
                            st.warning("No positive reviews found for this product.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col_ai3:
            if st.button("⭐ Top Products"):
                with st.spinner("Identifying customer favorites..."):
                    try:
                        top_products_analysis, usage = identify_top_products(st.session_state["df"], api_key)
                        st.markdown(top_products_analysis)
                        if usage:
                            st.session_state["total_tokens"] += usage.total_tokens
                            cost = (usage.prompt_tokens / 1000 * 0.03) + (usage.completion_tokens / 1000 * 0.06)
                            st.session_state["total_cost"] += cost
                            st.caption(f"Tokens used: {usage.total_tokens} (~${cost:.4f})")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col_ai4:
            if st.button("⚠️ Issues & Improvements"):
                with st.spinner("Analyzing negative feedback..."):
                    try:
                        negative_analysis, usage = analyze_negative_reviews(filtered_df, api_key)
                        st.markdown(negative_analysis)
                        if usage:
                            st.session_state["total_tokens"] += usage.total_tokens
                            cost = (usage.prompt_tokens / 1000 * 0.03) + (usage.completion_tokens / 1000 * 0.06)
                            st.session_state["total_cost"] += cost
                            st.caption(f"Tokens used: {usage.total_tokens} (~${cost:.4f})")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Sentiment breakdown
        if "SENTIMENT_CATEGORY" in st.session_state["df"].columns:
            st.subheader("📈 Sentiment Breakdown")
            sentiment_counts = filtered_df["SENTIMENT_CATEGORY"].value_counts()
            st.bar_chart(sentiment_counts)
        
        # Display cumulative API usage
        st.divider()
        col_usage1, col_usage2 = st.columns(2)
        with col_usage1:
            st.metric("Total Tokens Used", f"{st.session_state['total_tokens']:,}")
        with col_usage2:
            st.metric("Estimated Cost", f"${st.session_state['total_cost']:.4f}")
        st.caption("💡 Pricing based on GPT-4: $0.03/1K prompt tokens, $0.06/1K completion tokens")
    else:
        st.warning("⚠️ OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
