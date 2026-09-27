import os
import datetime as dt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Shopora RFM Segmentation", page_icon="📊", layout="wide")
st.title("📊 Customer RFM Segmentation & AI Marketing Copy")

CSV_PATH = "ecommerce_customer_behavior_dataset_v2.csv"

# --- Sidebar: data source + Groq API key ---
with st.sidebar:
    st.header("Settings")

    uploaded_file = st.file_uploader("Upload customer CSV (optional)", type=["csv"])

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password")

    n_clusters = st.slider("Number of KMeans clusters", 2, 8, 4)


# --- Load + clean data ---
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(file):
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Customer_ID"] = df["Customer_ID"].astype(str)
    df = df[df["Total_Amount"] > 0]
    return df


# --- Build RFM table + segments ---
@st.cache_data(show_spinner="Computing RFM segments...")
def build_rfm(df, n_clusters):
    snapshot_date = df["Date"].max() + dt.timedelta(days=1)

    rfm = df.groupby("Customer_ID").agg(
        {
            "Date": lambda x: (snapshot_date - x.max()).days,
            "Order_ID": "count",
            "Total_Amount": "sum",
        }
    ).reset_index()
    rfm.columns = ["Customer_ID", "Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1], duplicates="drop")
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4], duplicates="drop")
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4], duplicates="drop")

    rfm["RFM_Score"] = rfm[["R_Score", "F_Score", "M_Score"]].astype(str).agg("".join, axis=1)
    rfm["Segment"] = rfm["RFM_Score"].apply(segment_customer)

    return rfm


def segment_customer(score):
    if score in ["444", "443", "434", "344"]:
        return "Champions"
    elif score[1] in ["3", "4"] and score[2] in ["3", "4"]:
        return "Loyal Customers"
    elif score[2] == "4" and score[1] in ["1", "2"]:
        return "Big Spenders"
    elif score[0] == "1":
        return "At Risk"
    else:
        return "Regular"


def generate_marketing_copy(segment, api_key):
    client = Groq(api_key=api_key)

    prompt = f"""
Write a high-converting marketing email for an e-commerce brand.

Brand Name: Shopora
Customer Name: User
Target Segment: {segment}

Rules:
- Address customer as "User"
- Mention brand name "Shopora"
- Use CTA exactly: Shop now and grab the deal
- End exactly with:
Best regards,
The Shopora Team
- Include Subject line
- 150-200 words
- Persuasive tone

Format:

Subject: ...

Hi User,

...

Shop now and grab the deal
Best regards,
The Shopora Team
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a senior ecommerce marketing strategist."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=800,
        reasoning_effort="low",
    )
    content = response.choices[0].message.content
    if not content:
        # Reasoning models can burn the whole token budget on internal
        # reasoning and leave content empty - surface that clearly instead
        # of silently returning nothing.
        raise RuntimeError(
            "Model returned no content (likely ran out of tokens on internal "
            "reasoning). Try again, or raise max_tokens further."
        )
    return content


# --- Resolve data source: uploaded file, else bundled CSV ---
data_source = uploaded_file if uploaded_file is not None else CSV_PATH

if not uploaded_file and not os.path.exists(CSV_PATH):
    st.error(f"Couldn't find `{CSV_PATH}`. Upload a CSV in the sidebar to continue.")
    st.stop()

df = load_data(data_source)
rfm = build_rfm(df, n_clusters)

# --- Segment distribution ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Segment Distribution")
    st.dataframe(rfm["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers"))

with col2:
    st.subheader("Customer Segments")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Segment", ax=ax)
    ax.set_title("Customer Segments (Recency vs Monetary)")
    st.pyplot(fig)

with st.expander("📋 View full RFM table"):
    st.dataframe(rfm)

st.divider()

# --- Marketing email generation ---
st.subheader("✉️ Generate Marketing Email")

segments = sorted(rfm["Segment"].unique())
selected_segment = st.selectbox("Target segment", segments)

if st.button("Generate Marketing Email", type="primary"):
    if not api_key:
        st.error("Please provide a Groq API key in the sidebar (or set GROQ_API_KEY in your .env file).")
    else:
        with st.spinner("Generating email..."):
            try:
                email = generate_marketing_copy(selected_segment, api_key)
                st.markdown("### Generated Email")
                st.text_area("Email content", email, height=350)
            except Exception as e:
                st.error(f"Error generating email: {e}")