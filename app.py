import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Stock Return vs Volume", layout="wide")

st.title("📈 Return vs Volume Explorer")

# Sidebar
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL").upper()
n_days = st.slider("Select number of days", min_value=10, max_value=365, value=60)
n_clusters = st.slider("Select number of clusters", min_value=2, max_value=6, value=3)

# Fetch + Clean Data
@st.cache_data
def fetch_data(ticker, period_days):
    data = yf.download(ticker, period=f"{period_days}d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns]
    data = data[['Close', 'Volume']].dropna()
    data['Return'] = data['Close'].pct_change()
    data = data.dropna()
    data['Return_Pct'] = data['Return'] * 100
    data['Volume_M'] = data['Volume'] / 1e6
    return data

df = fetch_data(ticker, n_days)

# Clustering
features = df[['Return_Pct', 'Volume_M']].copy()
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

# Calculate medians for quadrant splitting
x_mid = df['Return_Pct'].median()
y_mid = df['Volume_M'].median()

# Assign quadrants
def classify_quadrant(row):
    if row['Return_Pct'] >= x_mid and row['Volume_M'] >= y_mid:
        return "High Return, High Volume"
    elif row['Return_Pct'] < x_mid and row['Volume_M'] >= y_mid:
        return "Low Return, High Volume"
    elif row['Return_Pct'] >= x_mid and row['Volume_M'] < y_mid:
        return "High Return, Low Volume"
    else:
        return "Low Return, Low Volume"

df['Quadrant'] = df.apply(classify_quadrant, axis=1)

# Plotly chart
fig = px.scatter(
    df,
    x='Return_Pct',
    y='Volume_M',
    color='Cluster',
    hover_data={'Return_Pct': ':.2f', 'Volume_M': ':.2f', 'Cluster': True, 'Date': df.index},
    color_continuous_scale='Viridis' if n_clusters > 2 else None,
    title=f"{ticker}: Daily Return vs Volume (last {n_days} days)",
    template='plotly_dark',
)

# Add quadrant lines
fig.add_vline(x=x_mid, line_dash="dash", line_color="gray")
fig.add_hline(y=y_mid, line_dash="dash", line_color="gray")

# Annotations for quadrants
annotations = [
    {"x": x_mid + 5, "y": y_mid + 5, "text": "📈 High Return\n📊 High Volume"},
    {"x": x_mid - 5, "y": y_mid + 5, "text": "📉 Low Return\n📊 High Volume"},
    {"x": x_mid + 5, "y": y_mid - 5, "text": "📈 High Return\n📉 Low Volume"},
    {"x": x_mid - 5, "y": y_mid - 5, "text": "📉 Low Return\n📉 Low Volume"},
]
for ann in annotations:
    fig.add_annotation(
        x=ann["x"], y=ann["y"], text=ann["text"],
        showarrow=False, font=dict(color="lightgray", size=12), opacity=0.7
    )

fig.update_layout(
    xaxis_title="Daily Return (%)",
    yaxis_title="Volume (Millions)",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=600,
    margin=dict(t=60, l=30, r=30, b=30),
)
fig.update_xaxes(ticksuffix="%")
fig.update_yaxes(ticksuffix="M")

# Show Plot
st.plotly_chart(fig, use_container_width=True)

# Stats table
st.subheader("📊 Cluster Summary Statistics")
summary = df.groupby('Cluster')[['Return_Pct', 'Volume_M']].agg(['mean', 'std', 'min', 'max']).round(2)
st.dataframe(summary)
