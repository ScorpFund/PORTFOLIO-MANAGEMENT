import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Stock Clustering", layout="wide")

# -----------------------------
# ✅ Data Fetching Function
# -----------------------------
@st.cache_data
def fetch_data(ticker: str, period_days: int) -> pd.DataFrame:
    try:
        data = yf.download(ticker, period=f"{period_days}d", progress=False)

        if data.empty:
            st.warning("❌ No data found. Please check the ticker symbol.")
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns]

        if not all(col in data.columns for col in ['Close', 'Volume']):
            st.warning("❌ Required columns ('Close', 'Volume') not found in data.")
            return pd.DataFrame()

        data = data[['Close', 'Volume']].dropna()
        data['Return'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        data['Return_Pct'] = data['Return'] * 100
        data['Volume_M'] = data['Volume'] / 1e6
        return data

    except Exception as e:
        st.error(f"⚠️ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# -----------------------------
# 📋 UI Components
# -----------------------------
st.title("📈 Volume vs. Return Clustering")

ticker = st.text_input("Enter stock ticker (e.g. AAPL):", value="AAPL").upper()
n_days = st.slider("Number of days to fetch:", 30, 365, 90, step=10)
n_clusters = st.slider("Number of clusters:", 2, 10, 4)

df = fetch_data(ticker, n_days)

if not df.empty:
    # -----------------------------
    # 🤖 Clustering
    # -----------------------------
    X = df[['Return_Pct', 'Volume_M']].copy()
    X_scaled = (X - X.mean()) / X.std()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # -----------------------------
    # ➗ Quadrant Classification
    # -----------------------------
    x_mid = df['Return_Pct'].median()
    y_mid = df['Volume_M'].median()

    def classify_quadrant(row):
        if row['Return_Pct'] >= x_mid and row['Volume_M'] >= y_mid:
            return '📈 High Volume, High Return'
        elif row['Return_Pct'] < x_mid and row['Volume_M'] >= y_mid:
            return '📉 High Volume, Low Return'
        elif row['Return_Pct'] >= x_mid and row['Volume_M'] < y_mid:
            return '📊 Low Volume, High Return'
        else:
            return '🔻 Low Volume, Low Return'

    df['Quadrant'] = df.apply(classify_quadrant, axis=1)

    # -----------------------------
    # 📊 Plotly Scatter Plot
    # -----------------------------
    fig = px.scatter(
        df,
        x='Return_Pct',
        y='Volume_M',
        color='Cluster',
        hover_data={'Return_Pct': ':.2f', 'Volume_M': ':.2f', 'Cluster': True, 'Quadrant': True},
        symbol='Quadrant',
        color_continuous_scale='Viridis',
        title=f"Clustering of {ticker} - Volume vs. Return"
    )

    # Add quadrant lines
    fig.add_shape(type='line', x0=x_mid, x1=x_mid, y0=df['Volume_M'].min(), y1=df['Volume_M'].max(),
                  line=dict(dash='dot', color='gray'))
    fig.add_shape(type='line', x0=df['Return_Pct'].min(), x1=df['Return_Pct'].max(), y0=y_mid, y1=y_mid,
                  line=dict(dash='dot', color='gray'))

    fig.update_layout(
        xaxis_title="Daily Return (%)",
        yaxis_title="Volume (Millions)",
        template="plotly_white",
        legend_title="Cluster",
        height=600
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 📈 Detailed Stats
    # -----------------------------
    st.subheader("📋 Quadrant Breakdown")
    st.dataframe(df['Quadrant'].value_counts().rename_axis("Quadrant").reset_index(name="Days"))

    st.subheader("📊 Cluster Statistics")
    cluster_stats = df.groupby('Cluster')[['Return_Pct', 'Volume_M']].agg(['mean', 'std', 'count'])
    st.dataframe(cluster_stats.style.format(precision=2))

else:
    st.info("👆 Enter a valid stock ticker and adjust sliders to see data.")
