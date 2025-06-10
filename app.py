import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv
import re

from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

load_dotenv()

@st.cache_data
def load_data():
    df = pd.read_csv("generated_5_year_budget_data.csv", parse_dates=['Date'])
    df['Variance'] = df['Actual_Spend'] - df['Allocated_Budget']
    df['Variance_%'] = (df['Variance'] / df['Allocated_Budget']) * 100
    return df

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

def detect_anomalies(df):
    df = df.copy()
    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict(df[['Actual_Spend']])
    return df[df['anomaly'] == -1]

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

st.set_page_config(page_title="💸 AI Financial Analysis", layout="wide")
st.title("🤖 AI-Powered Financial Analysis Dashboard")

try:
    df = load_data()
    st.sidebar.header("🔍 Filter")
    department = st.sidebar.selectbox("Select Department", df['Department'].unique())
    category = st.sidebar.selectbox("Select Category", df['Category'].unique())
    filtered = df[(df['Department'] == department) & (df['Category'] == category)]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📉 Variance",
        "📈 Forecasting",
        "🚨 Anomalies",
        "🧠 AI Risk & LLM Queries",
        "📊 Advanced Analytics"
    ])

    with tab1:
        st.subheader("📊 Department-wise Budget Summary")
        department_analysis = df.groupby('Department').agg({
            'Allocated_Budget': 'sum',
            'Actual_Spend': 'sum',
            'Variance': 'sum',
            'Variance_%': 'mean'
        }).reset_index()
        st.dataframe(department_analysis)

        fig = px.bar(
            department_analysis,
            x="Department",
            y="Variance",
            title="Total Variance by Department",
            color="Department",
            text_auto=".2s"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"📉 Variance Over Time - {department} / {category}")
        fig = px.line(filtered, x='Date', y='Variance', title='Variance Over Time')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📈 Spend Forecast (Prophet)")
        forecast = forecast_budget(df, department, category)
        if forecast is not None:
            fig_forecast = px.line(forecast, x='ds', y='yhat', title='Forecasted Spend')
            fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound')
            fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound')
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning("❗ Not enough data for forecasting.")

    with tab4:
        st.subheader("🚨 Anomaly Detection")
        anomalies = detect_anomalies(filtered)
        if not anomalies.empty:
            st.dataframe(anomalies[['Date', 'Actual_Spend', 'Allocated_Budget', 'Variance']])
            anomalies_clean = anomalies.dropna(subset=['Variance'])
            anomalies_clean = anomalies_clean[anomalies_clean['Variance'] > 0]
            fig_ano = px.scatter(anomalies_clean, x='Date', y='Actual_Spend', color='Category', size='Variance',
                                 title="Detected Anomalies by Spend and Date")
            st.plotly_chart(fig_ano, use_container_width=True)
        else:
            st.info("✅ No anomalies detected.")

    with tab5:
        st.subheader("⚠️ AI Risk Scores")
        risk_df = ai_risk_score(df)
        st.dataframe(risk_df[['Department', 'Category', 'Date', 'Variance_%', 'AI_Risk_Score']].head(10))

        st.subheader("🤖 Ask Anything (LLM-Powered Query)")
        st.markdown("""
        **💡 Ask in simple English like:**
        - Show total variance by department as a bar chart
        - Plot actual vs allocated spend per year
        - Show average efficiency for each category
        """)

        user_key = st.text_input("Enter your OpenAI API key", type="password")
        query = st.text_area("Ask your financial question:", placeholder="e.g., plot total spend by year")

        if query:
            if not user_key:
                st.error("❌ Please enter your OpenAI API key above.")
            else:
                try:
                    with st.spinner("🤖 Analyzing and generating chart..."):
                        llm = ChatOpenAI(
                            model_name="gpt-3.5-turbo",
                            temperature=0,
                            openai_api_key=user_key
                        )

                        system_message = (
                            "You are a financial data assistant. Use Plotly to generate all visualizations.\n"
                            "If you use matplotlib, always end with 'st.pyplot(plt)' instead of 'plt.show()'.\n"
                            "Your task is to analyze the dataframe provided and return a valid Streamlit-compatible Python code block."
                        )

                        agent = create_pandas_dataframe_agent(
                            llm=llm,
                            df=df.head(200),
                            agent_type=AgentType.OPENAI_FUNCTIONS,
                            verbose=True,
                            handle_parsing_errors=True,
                            allow_dangerous_code=True
                        )

                        st.markdown("**🔎 You asked:** " + query)
                        response = agent.run(f"{system_message}\n{query}")

                        code_match = re.search(r"```python\s+(.*?)```", response, re.DOTALL)
                        code = code_match.group(1) if code_match else response.strip()

                        # 🩹 Replace fig.show() with st.plotly_chart(fig)
                        if "fig.show()" in code:
                            code = code.replace("fig.show()", "st.plotly_chart(fig)")

                        st.code(code, language="python")
                        st.write("✅ Executing the code below:")
                        try:
                            exec(code, globals())
                        except Exception as e:
                            st.error(f"⚠️ Error executing code: {e}")
                except Exception as e:
                    st.error(f"⚠️ Error processing query: {e}")

    with tab6:
        st.subheader("📊 Advanced Financial Analytics")
        monthly_trend = df.resample('M', on='Date')['Actual_Spend'].sum().reset_index()
        fig_trend = px.line(monthly_trend, x='Date', y='Actual_Spend', title='Monthly Spend Trend')
        st.plotly_chart(fig_trend, use_container_width=True)

        df['Efficiency'] = df['Actual_Spend'] / df['Allocated_Budget']
        efficiency_by_dept = df.groupby('Department')['Efficiency'].mean().reset_index()
        fig_eff = px.bar(efficiency_by_dept, x='Department', y='Efficiency', title='Average Efficiency Score by Department')
        st.plotly_chart(fig_eff, use_container_width=True)

        df['Year'] = df['Date'].dt.year
        yoy = df.groupby(['Year', 'Category'])['Actual_Spend'].sum().reset_index()
        fig_yoy = px.line(yoy, x='Year', y='Actual_Spend', color='Category', markers=True,
                          title="Year-over-Year Category Spend")
        st.plotly_chart(fig_yoy, use_container_width=True)

        top_variance = df.groupby(['Department', 'Category'])['Variance'].sum().abs().reset_index()
        top_variance = top_variance.sort_values('Variance', ascending=False).head(5)
        fig_topvar = px.bar(top_variance, x='Variance', y='Department', color='Category', orientation='h',
                            title="Top 5 Variance Contributors")
        st.plotly_chart(fig_topvar, use_container_width=True)

        over_budget = df[df['Variance_%'] > 10]
        risk_summary = over_budget['Department'].value_counts().reset_index()
        risk_summary.columns = ['Department', 'Over_Budget_Count']
        fig_risk = px.bar(risk_summary, x='Department', y='Over_Budget_Count',
                          title="High Risk Departments (frequent over-budget)")
        st.plotly_chart(fig_risk, use_container_width=True)

except FileNotFoundError:
    st.error("❌ 'generated_5_year_budget_data.csv' not found in the current directory.")
