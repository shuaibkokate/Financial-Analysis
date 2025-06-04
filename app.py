import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# -------------------------------
# Load and prepare data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("generated_5_year_budget_data.csv", parse_dates=['Date'])
    df['Variance'] = df['Actual_Spend'] - df['Allocated_Budget']
    df['Variance_%'] = (df['Variance'] / df['Allocated_Budget']) * 100
    return df

# -------------------------------
# Forecasting using Prophet
# -------------------------------
def forecast_budget(df, department, category):
    filtered = df[(df['Department'] == department) & (df['Category'] == category)]
    timeseries = filtered.groupby("Date").sum().reset_index()[['Date', 'Actual_Spend']]
    timeseries.columns = ['ds', 'y']

    if len(timeseries) < 2:
        return None

    model = Prophet()
    model.fit(timeseries)
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# -------------------------------
# Anomaly Detection
# -------------------------------
def detect_anomalies(df):
    df = df.copy()
    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict(df[['Actual_Spend']])
    return df[df['anomaly'] == -1]

# -------------------------------
# AI Risk Scoring
# -------------------------------
def ai_risk_score(df):
    df = df.copy()
    le_dept = LabelEncoder()
    le_cat = LabelEncoder()
    df['DeptCode'] = le_dept.fit_transform(df['Department'])
    df['CatCode'] = le_cat.fit_transform(df['Category'])

    X = df[['Allocated_Budget', 'Actual_Spend', 'DeptCode', 'CatCode']]
    y = (df['Variance_%'] > 10).astype(int)

    model = RandomForestRegressor()
    model.fit(X, y)
    df['AI_Risk_Score'] = model.predict(X)

    return df.sort_values("AI_Risk_Score", ascending=False)

# -------------------------------
# NLP-like Query Parser
# -------------------------------
def handle_query(query, df):
    q = query.lower()
    if "over budget" in q:
        return df[df['Variance'] > 0]
    elif "under budget" in q:
        return df[df['Variance'] < 0]
    elif "anomalies" in q:
        return detect_anomalies(df)
    elif "high risk" in q:
        return ai_risk_score(df).head(10)
    else:
        return "❌ Sorry, I didn't understand your query."

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="💸 AI Financial Analysis", layout="wide")
st.title("🤖 AI-Powered Financial Analysis Dashboard")

try:
    df = load_data()

    # Sidebar Filters
    st.sidebar.header("🔍 Filter")
    department = st.sidebar.selectbox("Select Department", df['Department'].unique())
    category = st.sidebar.selectbox("Select Category", df['Category'].unique())
    filtered = df[(df['Department'] == department) & (df['Category'] == category)]

    # Tabs for features
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📉 Variance", "📈 Forecasting", "🚨 Anomalies", "⚠️ AI Risk & NLP"])

    with tab1:
        st.subheader("📊 Full Data Overview")
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader(f"📉 Variance Over Time - {department} / {category}")
        fig = px.line(filtered, x='Date', y='Variance', title='Variance Over Time')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📈 Spend Forecast (Prophet)")
        forecast = forecast_budget(df, department, category)
        if forecast is not None:
            fig_forecast = px.line(forecast, x='ds', y='yhat', title='Forecasted Spend')
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning("❗ Not enough data for forecasting.")

    with tab4:
        st.subheader("🚨 Anomaly Detection")
        anomalies = detect_anomalies(filtered)
        if not anomalies.empty:
            st.dataframe(anomalies[['Date', 'Actual_Spend', 'Allocated_Budget', 'Variance']])
        else:
            st.info("✅ No anomalies detected.")

    with tab5:
        st.subheader("⚠️ AI-Based Risk Scores")
        risk_df = ai_risk_score(df)
        st.dataframe(risk_df[['Department', 'Category', 'Date', 'Variance_%', 'AI_Risk_Score']].head(10))

        st.subheader("💬 Ask AI a Question")
        query = st.text_input("Examples: 'Show over budget', 'Show high risk', 'Show anomalies'")
        if query:
            result = handle_query(query, df)
            st.write(result)

except FileNotFoundError:
    st.error("❌ 'generated_5_year_budget_data.csv' not found in the current directory.")
