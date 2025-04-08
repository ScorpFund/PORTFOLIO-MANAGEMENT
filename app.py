import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Stock Return vs Volume", layout="wide")

# ---------- Sidebar Inputs ----------
st.sidebar.title("📊 Stock Analyzer")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
n_days = st.sidebar.slider("Number of Days", 10, 365, 90)
n_clusters = st.sidebar.slider("Number of Clusters", 1, 6, 3)

# ---------- Fetch & Prepare Data ----------
@st.cache_data
def fetch_data(ticker, period_days):
    data = yf.download(ticker, period=f"{period_days}d", progress=False)

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns]

    if 'Close' not in data.columns or 'Volume' not in data.columns:
        return pd.DataFrame()

    data = data[['Close', 'Volume']].dropna()
    data['Return'] = data['Close'].pct_change()
    data = data.dropna()
    data['Return_Pct'] = data['Return'] * 100  # Convert to percentage
    data['Volume_M'] = data['Volume'] / 1e6    # Convert to millions
    return data

df = fetch_data(ticker, n_days)

if df.empty:
    st.error("❌ No data found. Please check the ticker symbol.")
    st.stop()

# ---------- Clustering ----------
features = df[['Return_Pct', 'Volume_M']]
kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

# ---------- Quadrant Classification ----------
x_mid = df['Return_Pct'].median()
y_mid = df['Volume_M'].median()

def classify_quadrant(row):
    if row['Return_Pct'] >= x_mid and row['Volume_M'] >= y_mid:
        return "High Return / High Volume"
    elif row['Return_Pct'] < x_mid and row['Volume_M'] >= y_mid:
        return "Low Return / High Volume"
    elif row['Return_Pct'] >= x_mid and row['Volume_M'] < y_mid:
        return "High Return / Low Volume"
    else:
        return "Low Return / Low Volume"

df['Quadrant'] = df.apply(classify_quadrant, axis=1)

# ---------- Plotly Scatter Plot ----------
fig = px.scatter(
    df,
    x="Return_Pct",
    y="Volume_M",
    color="Cluster",
    hover_data=["Close", "Volume", "Return_Pct", "Volume_M"],
    labels={
        "Return_Pct": "Daily Return (%)",
        "Volume_M": "Volume (Millions)"
    },
    title=f"{ticker} - Return vs Volume",
)

fig.update_layout(
    template="plotly_white",
    title_font_size=20,
    title_font_color="#222",
    plot_bgcolor="#f9f9f9",
    xaxis=dict(showgrid=True, zeroline=True),
    yaxis=dict(showgrid=True, zeroline=True),
)

# Add quadrant lines
fig.add_shape(type="line", x0=x_mid, x1=x_mid, y0=df['Volume_M'].min(), y1=df['Volume_M'].max(),
              line=dict(color="gray", width=1, dash="dot"))
fig.add_shape(type="line", x0=df['Return_Pct'].min(), x1=df['Return_Pct'].max(), y0=y_mid, y1=y_mid,
              line=dict(color="gray", width=1, dash="dot"))

st.plotly_chart(fig, use_container_width=True)

# ---------- Stats Below Plot ----------
st.subheader("📈 Cluster & Quadrant Stats")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔍 Cluster Breakdown")
    st.dataframe(df.groupby("Cluster")[["Return_Pct", "Volume_M"]].mean().round(2))

with col2:
    st.markdown("### 🧭 Quadrant Distribution")
    st.dataframe(df["Quadrant"].value_counts().reset_index().rename(columns={"index": "Quadrant", "count": "Count"}))
