import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Clustering App", layout="wide")
st.title("📈 Stock Clustering by Volume & Return")

# Sidebar controls
tickers = st.sidebar.multiselect(
    "Select Tickers",
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'IBM', 'ORCL'],
    default=['AAPL', 'MSFT', 'GOOGL']
)
period = st.sidebar.selectbox("Select Period", ['1mo', '3mo', '6mo', '1y'], index=2)
clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=6, value=3)
log_scale = st.sidebar.checkbox("Use Log Scale for Volume (x-axis)", value=True)

@st.cache_data
def fetch_data(tickers, period):
    data = yf.download(tickers, period=period, progress=False, group_by="ticker")
    
    if isinstance(tickers, str) or len(tickers) == 1:
        if isinstance(tickers, list):
            tickers = tickers[0]
        prices = data['Adj Close'] if 'Adj Close' in data else data['Close']
        volumes = data['Volume']
        prices = pd.DataFrame({tickers: prices})
        volumes = pd.DataFrame({tickers: volumes})
    else:
        prices = data['Adj Close']
        volumes = data['Volume']
    
    return prices, volumes

if tickers:
    with st.spinner("Fetching stock data..."):
        prices, volumes = fetch_data(tickers, period)

    # Compute daily returns
    returns = prices.pct_change().dropna()
    avg_returns = returns.mean()
    avg_volumes = volumes.mean()

    # Combine into DataFrame
    raw_features = pd.DataFrame({
        'avg_volume': avg_volumes,
        'avg_return': avg_returns
    })

    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(raw_features)

    # KMeans clustering
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    raw_features['cluster'] = kmeans.fit_predict(scaled_features)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in sorted(raw_features['cluster'].unique()):
        cluster = raw_features[raw_features['cluster'] == label]
        x = cluster['avg_volume']
        y = cluster['avg_return']
        ax.scatter(x, y, label=f"Cluster {label}", s=100)
        for ticker in cluster.index:
            ax.text(x[ticker], y[ticker], ticker)

    ax.set_xlabel("Average Volume (log scale)" if log_scale else "Average Volume")
    ax.set_ylabel("Average Return")
    ax.set_title("Stock Clusters by Normalized Volume & Return")
    ax.legend()
    ax.grid(True)

    if log_scale:
        ax.set_xscale("log")

    st.pyplot(fig)

    with st.expander("📋 Cluster Data Table"):
        st.dataframe(
            raw_features.style.format({
                "avg_volume": "{:,.0f}",
                "avg_return": "{:.2%}"
            })
        )
else:
    st.info("Please select at least one ticker from the sidebar.")
