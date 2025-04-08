import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

def load_data(ticker, n_days):
    try:
        data = yf.download(ticker, period=f"{n_days}d", progress=False, 
auto_adjust=False)
        if data.empty:
            raise Exception("No data found")
        # Perform data cleaning and transformation
        data['Return'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        data['Return_Pct'] = data['Return'] * 100
        data['Volume_M'] = data['Volume'] / 1e6
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def cluster_data(data, n_clusters):
    try:
        features = data[['Return_Pct', 'Volume_M']]
        features_scaled = (features - features.mean()) / features.std()
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        data['Cluster'] = kmeans.fit_predict(features_scaled)
        return data
    except Exception as e:
        print(f"Error clustering data: {str(e)}")
        return None

def visualize_data(data):
    try:
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
        # Add shapes for x and y midpoints
        fig.add_shape(type='line', x0=data['Return_Pct'].median(), 
x1=data['Return_Pct'].median(), y0=data['Volume_M'].median(), 
y1=data['Volume_M'].median(),
                      line=dict(color='gray', dash='dot'))
        fig.add_shape(type='line', x0=data['Return_Pct'].min(), 
x1=data['Return_Pct'].max(), y0=data['Volume_M'].min(), 
y1=data['Volume_M'].max(),
                      line=dict(color='gray', dash='dot'))

        fig.update_layout(
            xaxis_title="Daily Return (%)",
            yaxis_title="Volume (Millions)",
            legend_title="Cluster"
        )
        return fig
    except Exception as e:
        print(f"Error visualizing data: {str(e)}")
        return None

def main():
    st.title("Stock Return vs Volume Clustering")

    # Input parameters
    ticker = st.text_input("Ticker Symbol", "")
    n_days = st.number_input("Number of Days", 1, 365)

    if not ticker or not n_days:
        st.error("Please enter a valid ticker symbol and number of days")
        return

    data = load_data(ticker, n_days)
    if not data:
        st.error("Failed to load data. Please try again later.")
        return

    data = cluster_data(data, 3)  # Use 3 clusters for now
    if not data:
        st.error("Failed to cluster data. Please try again later.")
        return

    fig = visualize_data(data)
    if not fig:
        st.error("Failed to visualize data. Please try again later.")
        return

    st.plotly_chart(fig, use_container_width=True)

    # Display quadrant breakdown
    st.markdown("### Quadrant Breakdown")
    st.dataframe(data['Quadrant'].value_counts().rename_axis("Quadrant").rest.dataframe(data['Quadrant'].value_counts().rename_axis("Quadrant").reset_index(name="Days")))

    # Display cluster stats
    st.markdown("### Cluster Stats")
    cluster_stats = data.groupby('Cluster')[['Return_Pct', 
'Volume_M']].agg(['mean', 'std', 'count']).round(2)
    st.dataframe(cluster_stats)

if __name__ == "__main__":
    main()
```
