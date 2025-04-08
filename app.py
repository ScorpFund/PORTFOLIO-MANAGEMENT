import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page setup
st.set_page_config(page_title="Stock Volume vs Return Clustering", layout="wide")
st.title("📊 Stock Return vs Volume Clustering")

# Sidebar input
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
num_days = st.sidebar.slider("Select Number of Days", min_value=10, max_value=180, value=60)
num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=6, value=3)

@st.cache_data
def fetch_data(ticker, period_days):
    data = yf.download(ticker, period=f"{period_days}d", progress=False)
    data = data[['Close', 'Volume']].dropna()
    data['Return'] = data['Close'].pct_change()
    data = data.dropna()
    data['Return_Pct'] = data['Return'] * 100
    data['Volume_M'] = data['Volume'] / 1e6
    return data

if ticker:
    with st.spinner("Fetching data..."):
        df = fetch_data(ticker, num_days)

    # Debug output
    st.subheader("🧪 Debug: Data Preview")
    st.dataframe(df.head())
    st.write("Columns:", df.columns.tolist())

    if all(col in df.columns for col in ['Return_Pct', 'Volume_M']) and not df[['Return_Pct', 'Volume_M']].isnull().any().any():
        # Clustering
        features = df[['Return_Pct', 'Volume_M']]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled).astype(str)

        # Midpoints for quadrants
        x_mid = df['Return_Pct'].median()
        y_mid = df['Volume_M'].median()

        try:
            # Plotly scatter plot
            fig = px.scatter(
                df,
                x='Return_Pct',
                y='Volume_M',
                color='Cluster',
                hover_data=['Close', 'Volume'],
                text=df.index.strftime('%Y-%m-%d'),
                title=f"{ticker} – Return vs Volume Clustering",
                template='simple_white'
            )

            # Quadrant lines
            fig.add_shape(type="line", x0=x_mid, x1=x_mid,
                          y0=df['Volume_M'].min(), y1=df['Volume_M'].max(),
                          line=dict(dash="dash", color="gray"))
            fig.add_shape(type="line", y0=y_mid, y1=y_mid,
                          x0=df['Return_Pct'].min(), x1=df['Return_Pct'].max(),
                          line=dict(dash="dash", color="gray"))

            # Labels
            fig.add_annotation(text="📈 High Vol / High Ret", x=x_mid + 5, y=y_mid + 5, showarrow=False)
            fig.add_annotation(text="📉 High Vol / Low Ret", x=x_mid - 5, y=y_mid + 5, showarrow=False)
            fig.add_annotation(text="📈 Low Vol / High Ret", x=x_mid + 5, y=y_mid - 5, showarrow=False)
            fig.add_annotation(text="📉 Low Vol / Low Ret", x=x_mid - 5, y=y_mid - 5, showarrow=False)

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

        except Exception as e:
            st.error(f"❌ Plotly error: {e}")
            st.dataframe(df.head())

        # Quadrant classification
        def classify_quadrant(row):
            if row['Return_Pct'] >= x_mid and row['Volume_M'] >= y_mid:
                return "High Vol / High Ret"
            elif row['Return_Pct'] < x_mid and row['Volume_M'] >= y_mid:
                return "High Vol / Low Ret"
            elif row['Return_Pct'] >= x_mid and row['Volume_M'] < y_mid:
                return "Low Vol / High Ret"
            else:
                return "Low Vol / Low Ret"

        df['Quadrant'] = df.apply(classify_quadrant, axis=1)

        st.subheader("📋 Summary Statistics by Quadrant")
        summary = df.groupby('Quadrant').agg(
            Count=('Return_Pct', 'count'),
            Avg_Return_Pct=('Return_Pct', 'mean'),
            Avg_Volume_M=('Volume_M', 'mean'),
            Min_Return=('Return_Pct', 'min'),
            Max_Return=('Return_Pct', 'max')
        ).round(2)

        st.dataframe(summary)

    else:
        st.error("❌ Required data columns missing or contain NaNs.")
else:
    st.info("Please enter a ticker symbol to begin.")
