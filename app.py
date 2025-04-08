import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

# --- App Title ---
st.set_page_config(page_title="Stock Return vs Volume", layout="centered")
st.title("📈 Return vs Volume Clustering")

# --- Sidebar Inputs ---
ticker = st.text_input("Enter Stock Ticker:", value="AAPL")
n_days = st.slider("Select Number of Days:", min_value=30, max_value=180, value=90)
n_clusters = st.slider("Select Number of Clusters:", min_value=2, max_value=6, value=4)

@st.cache_data
def fetch_data(ticker, n_days):
    df = yf.download(ticker, period=f"{n_days}d", progress=False, auto_adjust=False)
    return df

# --- Fetch Data ---
data = fetch_data(ticker, n_days)

if data.empty:
    st.error("❌ No data found. Please check the ticker symbol.")
else:
    data = data[['Close', 'Volume']].dropna()
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    data['Return_Pct'] = data['Return'] * 100
    data['Volume_M'] = data['Volume'] / 1e6

    # Filter outliers
    data = data[(data['Return_Pct'].between(-50, 50)) & (data['Volume_M'] > 0)]

    # Normalize for clustering
    features = data[['Return_Pct', 'Volume_M']]
    features_scaled = (features - features.mean()) / features.std()

    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    data['Cluster'] = kmeans.fit_predict(features_scaled)

    # Midpoints
    x_mid = data['Return_Pct'].median()
    y_mid = data['Volume_M'].median()

    # Quadrant classification
    def classify_quadrant(row):
        return_pct = row.get('Return_Pct')  # Use .get() to handle potential missing keys
        volume_m = row.get('Volume_M')      # Use .get()

        if pd.notna(return_pct) and pd.notna(volume_m): # Check for nulls
            if return_pct >= x_mid and volume_m >= y_mid:
                return '📈 High Volume, High Return'
            elif return_pct < x_mid and volume_m >= y_mid:
                return '📉 High Volume, Low Return'
            elif return_pct >= x_mid and volume_m < y_mid:
                return '📊 Low Volume, High Return'
            else:
                return '🔻 Low Volume, Low Return'
        else:
            return '❓ Data Missing' # Explicitly handle missing data

    data['Quadrant'] = data.apply(classify_quadrant, axis=1)

    # Plotly Scatter
    fig = px.scatter(
        data,
        x='Return_Pct',
        y='Volume_M',
        color=data['Cluster'].astype(str),
        symbol='Quadrant',
        hover_data={'Return_Pct': ':.1f', 'Volume_M': ':.2f'},
        title=f"{ticker.upper()} Daily Return vs Volume (Last {n_days} Days)",
        template='plotly_white',
        height=600
    )

    fig.add_shape(type='line', x0=x_mid, x1=x_mid, y0=data['Volume_M'].min(), y1=data['Volume_M'].max(),
                  line=dict(color='gray', dash='dot'))
    fig.add_shape(type='line', x0=data['Return_Pct'].min(), x1=data['Return_Pct'].max(), y0=y_mid, y1=y_mid,
                  line=dict(color='gray', dash='dot'))

    fig.update_layout(
        xaxis_title="Daily Return (%)",
        yaxis_title="Volume (Millions)",
        legend_title="Cluster",
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Stats Section ---
    st.subheader("📌 Quadrant Breakdown")
    st.dataframe(data['Quadrant'].value_counts().rename_axis("Quadrant").reset_index(name="Days"))

    st.subheader("📌 Cluster Statistics")
    stats = data.groupby('Cluster')[['Return_Pct', 'Volume_M']].agg(['mean', 'std', 'count']).round(2)
    st.dataframe(stats)
