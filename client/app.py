import streamlit as st
import pandas as pd
import requests

API_URL = "https://tradesense-api-d1iu.onrender.com"  # change to server IP when deploying

st.title("Investor Behavioral Bias Detection System")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Raw Data Preview")
    st.dataframe(df.head())

    # Column mapping UI — unchanged
    investor_col = st.selectbox("Investor ID column", df.columns)
    date_col = st.selectbox("Date column", df.columns)
    action_col = st.selectbox("Action column (Buy/Sell)", df.columns)
    value_col = st.selectbox("Transaction Value column", df.columns)
    units_col = st.selectbox("Units column", df.columns)

    df = df.rename(columns={
        investor_col: "investor_id",
        date_col: "date",
        action_col: "action",
        value_col: "total_value",
        units_col: "units"
    })

    # Send to API instead of running models locally
    with st.spinner("Analyzing investor behavior..."):
        payload = {
            "transactions": df[["investor_id", "date", "action", "total_value", "units"]].to_dict(orient="records")
        }
        response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        features = pd.DataFrame(response.json()).set_index("investor_id")

        num_investors = len(features)

        # UI — exactly the same as your existing code
        if num_investors == 1:
            investor_id = features.index[0]
            st.subheader("Individual Behavioral Risk Report")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Behavioral Risk Score",
                    f"{features['risk_score'].iloc[0]:.1f}"
                )
            with col2:
                st.metric(
                    "Dominant Bias",
                    features["dominant_bias"].iloc[0]
                )
            with col3:
                st.metric(
                    "Risk Category",
                    features["risk_category"].iloc[0]
                )

            st.subheader("Behavior Breakdown")
            bias_df = pd.DataFrame({
                "Bias": ["Overtrading", "Panic Selling", "Short-Term"],
                "Probability": [
                    features["overtrading_prob"].iloc[0],
                    features["panic_prob"].iloc[0],
                    features["short_prob"].iloc[0]
                ]
            })
            st.bar_chart(bias_df.set_index("Bias"))

        else:
            st.subheader("Behavioral Risk Results")
            st.dataframe(features)

            st.subheader("Risk Score Distribution")
            st.bar_chart(features["risk_score"])

            st.subheader("Dominant Bias Distribution")
            st.bar_chart(features["dominant_bias"].value_counts())

            st.subheader("Risk Category Distribution")
            st.bar_chart(features["risk_category"].value_counts())

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Investors", len(features))
            with col2:
                st.metric(
                    "High Risk Investors",
                    (features["risk_score"] > 70).sum()
                )
            with col3:
                st.metric(
                    "No Significant Bias",
                    (features["dominant_bias"] == "No Significant Bias").sum()
                )

    else:
        st.error(f"API error {response.status_code} — {response.text}")