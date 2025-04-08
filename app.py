# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📈 Stock Return vs Volume Clustering")

# --- Sidebar Inputs ---
st.sidebar.header("Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
n_days = st.sidebar.slider("Number of Days", min_value=30, max_value=180, value=90, step=10)
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=6, value=4, step=1)

# --- Fetch and Process Data ---
@st.cache_data(show_spinner=False)
def load_data(ticker, n_days):
    data = yf.download(ticker, period=f"{n_days}d", progress=False, auto_adjust=False)
    if data.empty:
        return None
    data = data[['Close', 'Volume']].dropna()
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    data['Return_Pct'] = data['Return'] * 100
    data['Volume_M'] = data['Volume'] / 1e6
    data = data[(data['Return_Pct'].between(-50, 50)) & (data['Volume_M'] > 0)]
    return data

data = load_data(ticker, n_days)

if data is None:
    st.error("❌ No data found. Please check the ticker symbol.")
else:
    features = data[['Return_Pct', 'Volume_M']]
    features_scaled = (features - features.mean()) / features.std()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    data['Cluster'] = kmeans.fit_predict(features_scaled)

    x_mid = data['Return_Pct'].median()
    y_mid = data['Volume_M'].median()

    def classify_quadrant(row):
        if float(row['Return_Pct']) >= x_mid and float(row['Volume_M']) >= y_mid:
            return '📈 High Volume, High Return'
        elif float(row['Return_Pct']) < x_mid and float(row['Volume_M']) >= y_mid:
            return '📉 High Volume, Low Return'
        elif float(row['Return_Pct']) >= x_mid and float(row['Volume_M']) < y_mid:
            return '📊 Low Volume, High Return'
        else:
            return '🔻 Low Volume, Low Return'

    data['Quadrant'] = data.apply(classify_quadrant, axis=1)

    # ✅ Validate expected columns before plotting
required_cols = ['Return_Pct', 'Volume_M', 'Cluster', 'Quadrant']
missing_cols = [col for col in required_cols if col not in data.columns or data[col].isnull().all()]
if missing_cols:
    st.error(f"Missing or empty required columns: {missing_cols}")
else:
    st.subheader(f"{ticker} Daily Returns vs Volume (Last {n_days} Days)")
    fig = px.scatter(
        data,
        x='Return_Pct',
        y='Volume_M',
        color='Cluster',
        symbol='Quadrant',
        hover_data=['Return_Pct', 'Volume_M'],
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

    # --- Plot ---
    st.subheader(f"{ticker} Daily Returns vs Volume (Last {n_days} Days)")
    fig = px.scatter(
        data,
        x='Return_Pct',
        y='Volume_M',
        color='Cluster',
        symbol='Quadrant',
        hover_data=['Return_Pct', 'Volume_M'],
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

    # --- Stats ---
    st.markdown("### 📌 Quadrant Breakdown")
    st.dataframe(data['Quadrant'].value_counts().rename_axis("Quadrant").reset_index(name="Days"))

    st.markdown("### 📌 Cluster Stats")
    cluster_stats = data.groupby('Cluster')[['Return_Pct', 'Volume_M']].agg(['mean', 'std', 'count']).round(2)
    st.dataframe(cluster_stats)
