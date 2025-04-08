import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st

# ---------------------- Data Fetching ----------------------
def fetch_stock_data(ticker, n_days):
    try:
        df = yf.download(ticker, period=f"{n_days}d", progress=False, auto_adjust=False)
        if df.empty:
            st.error(f"❌ No data found for '{ticker}' over the last {n_days} days.")
            return None
        return df
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return None

# ---------------------- Data Processing ----------------------
def process_stock_data(df):
    df = df[['Close', 'Volume']].dropna()
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    df['Return_Pct'] = df['Return'] * 100
    df['Volume_M'] = df['Volume'] / 1e6

    # Handle outliers
    z_return = np.abs((df['Return_Pct'] - df['Return_Pct'].mean()) / df['Return_Pct'].std())
    z_volume = np.abs((df['Volume_M'] - df['Volume_M'].mean()) / df['Volume_M'].std())
    df = df[(z_return < 3) & (z_volume < 3)]

    return df

# ---------------------- Clustering ----------------------
def perform_clustering(df, n_clusters):
    features = df[['Return_Pct', 'Volume_M']].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = (features - features.mean()) / features.std()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled)
    return df

# ---------------------- Quadrant Classification ----------------------
def classify_quadrants(df):
    x_mid = float(df['Return_Pct'].median())
    y_mid = float(df['Volume_M'].median())

    def get_quadrant(row):
        x = float(row['Return_Pct'])
        y = float(row['Volume_M'])

        if (x >= x_mid) and (y >= y_mid):
            return '📈 High Volume, High Return'
        elif (x < x_mid) and (y >= y_mid):
            return '📉 High Volume, Low Return'
        elif (x >= x_mid) and (y < y_mid):
            return '📊 Low Volume, High Return'
        else:
            return '🔻 Low Volume, Low Return'

    df['Quadrant'] = df.apply(get_quadrant, axis=1)
    return df

# ---------------------- Visualization ----------------------

def plot_cluster_scatter(df, ticker, n_days):
    fig = px.scatter(
        df,
        x="Return_Pct",
        y="Volume_M",
        color="Cluster",  # ← Fix: pass the column name, not a Series
        symbol="Quadrant",
        hover_data=["Return_Pct", "Volume_M"],
        title=f"{ticker} Daily Return vs Volume (Last {n_days} Days)",
        template="plotly_white",
        height=600
    )

    # Add quadrant lines
    fig.add_shape(
        type="line",
        x0=df["Return_Pct"].median(), x1=df["Return_Pct"].median(),
        y0=df["Volume_M"].min(), y1=df["Volume_M"].max(),
        line=dict(color="gray", dash="dot")
    )
    fig.add_shape(
        type="line",
        x0=df["Return_Pct"].min(), x1=df["Return_Pct"].max(),
        y0=df["Volume_M"].median(), y1=df["Volume_M"].median(),
        line=dict(color="gray", dash="dot")
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Stats Display ----------------------
def display_cluster_statistics(df):
    st.subheader("📌 Cluster Statistics")
    try:
        stats = (
            df.groupby("Cluster")
              .agg({
                  "Return_Pct": ['mean', 'std', 'count'],
                  "Volume_M": ['mean', 'std', 'count']
              })
        )
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        stats = stats.round(2)
        st.dataframe(stats)
    except Exception as e:
        st.error(f"⚠️ Error computing cluster statistics: {e}")

def display_quadrant_breakdown(df):
    st.subheader("📌 Quadrant Breakdown")
    breakdown = (
        df["Quadrant"]
          .value_counts()
          .rename_axis("Quadrant")
          .reset_index(name="Days")
    )
    st.dataframe(breakdown)

# ---------------------- Main App ----------------------
def main():
    st.set_page_config(page_title="Stock Cluster Analyzer", layout="wide")
    st.title("📊 Stock Return & Volume Cluster Analysis")

    # Sidebar input
    with st.sidebar:
        ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
        n_days = st.slider("Days of Data", 30, 180, 90)
        n_clusters = st.slider("Number of Clusters", 2, 6, 4)

    df = fetch_stock_data(ticker, n_days)
    if df is not None:
        df = process_stock_data(df)
        df = perform_clustering(df, n_clusters)
        df = classify_quadrants(df)

        plot_cluster_scatter(df, ticker, n_days)
        display_quadrant_breakdown(df)
        display_cluster_statistics(df)
    else:
        st.info("Please enter a valid ticker and try again.")

# ---------------------- Run ----------------------
if __name__ == "__main__":
    main()
