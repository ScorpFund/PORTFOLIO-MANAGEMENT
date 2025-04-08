import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# App layout
st.set_page_config(page_title="Stock Volume vs Return", layout="wide")
st.title("📊 Daily Stock Return vs Volume Clustering")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
num_days = st.sidebar.slider("Select Number of Days", min_value=10, max_value=180, value=60)
num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=6, value=3)

@st.cache_data
def fetch_data(ticker, period_days):
    df = yf.download(ticker, period=f"{period_days}d", progress=False)
    df = df[['Close', 'Volume']].dropna()
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()
    df['Volume_M'] = df['Volume'] / 1e6
    df['Return_Pct'] = df['Return'] * 100
    return df

if ticker:
    with st.spinner("Fetching data..."):
        df = fetch_data(ticker, num_days)

    # Normalize and cluster
    features = df[['Return_Pct', 'Volume_M']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Plot
    fig = px.scatter(
        df,
        x='Return_Pct',
        y='Volume_M',
        color='Cluster',
        hover_data=['Close', 'Volume', 'Return_Pct'],
        text=df.index.strftime('%Y-%m-%d'),
        title=f"{ticker} – Return vs Volume Clustering",
        template='simple_white'
    )

    # Quadrants
    x_mid = df['Return_Pct'].median()
    y_mid = df['Volume_M'].median()
    fig.add_shape(type="line", x0=x_mid, x1=x_mid, y0=df['Volume_M'].min(), y1=df['Volume_M'].max(),
                  line=dict(dash="dash", color="gray"))
    fig.add_shape(type="line", y0=y_mid, y1=y_mid, x0=df['Return_Pct'].min(), x1=df['Return_Pct'].max(),
                  line=dict(dash="dash", color="gray"))

    # Labels for quadrants
    fig.add_annotation(text="📈 High Vol / High Ret", x=x_mid + 5, y=y_mid + 5, showarrow=False, font_size=12)
    fig.add_annotation(text="📉 High Vol / Low Ret", x=x_mid - 5, y=y_mid + 5, showarrow=False, font_size=12)
    fig.add_annotation(text="📈 Low Vol / High Ret", x=x_mid + 5, y=y_mid - 5, showarrow=False, font_size=12)
    fig.add_annotation(text="📉 Low Vol / Low Ret", x=x_mid - 5, y=y_mid - 5, showarrow=False, font_size=12)

    fig.update_traces(marker=dict(size=10), textposition="top center")
    fig.update_layout(
        xaxis_title="Daily Return (%)",
        yaxis_title="Volume (Millions)",
        height=600,
        font=dict(size=14),
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(bgcolor="white", font_size=13)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 📊 Stats Table
    st.subheader("📋 Summary Statistics")

    def quadrant_label(row):
        return (
            "Top Right" if row['Return_Pct'] >= x_mid and row['Volume_M'] >= y_mid else
            "Top Left" if row['Return_Pct'] < x_mid and row['Volume_M'] >= y_mid else
            "Bottom Right" if row['Return_Pct'] >= x_mid and row['Volume_M'] < y_mid else
            "Bottom Left"
        )

    df['Quadrant'] = df.apply(quadrant_label, axis=1)

    summary = df.groupby('Quadrant').agg(
        Count=('Return_Pct', 'count'),
        Avg_Return_Pct=('Return_Pct', 'mean'),
        Avg_Volume_M=('Volume_M', 'mean'),
        Min_Return=('Return_Pct', 'min'),
        Max_Return=('Return_Pct', 'max')
    ).round(2).sort_index()

    st.dataframe(summary)

else:
    st.info("Please enter a valid ticker symbol in the sidebar.")
