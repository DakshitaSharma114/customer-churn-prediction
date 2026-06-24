

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load model
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Page settings
st.set_page_config(
    page_title="AI Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🤖 Prediction", "📈 Model Performance"]
)

# =========================
# OVERVIEW PAGE
# =========================

if page == "🏠 Overview":

    st.title("📊 AI Customer Churn Analytics Platform")

    st.write("Analyze customer churn risk using Machine Learning")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Customers", "7032")

    with col2:
        st.metric("Churn Rate", "26.5%")

    with col3:
        st.metric("Model Accuracy", "78.8%")

    with col4:
        st.metric("Features", "30")

    st.divider()

    st.subheader("📌 Key Business Insights")

    st.info("""
• Month-to-month customers churn the most

• New customers are more likely to leave

• Higher monthly charges increase churn risk

• Long-term contracts reduce churn significantly
""")

# =========================
# PREDICTION PAGE
# =========================

elif page == "🤖 Prediction":

    st.title("🤖 Customer Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        monthly_charges = st.slider(
            "Monthly Charges",
            min_value=0.0,
            max_value=150.0,
            value=50.0
        )

    with col2:
        contract = st.selectbox(
            "Contract Type",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    if st.button("🚀 Predict Churn Risk"):

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=model_columns
        )

        if "tenure" in input_data.columns:
            input_data["tenure"] = tenure

        if "MonthlyCharges" in input_data.columns:
            input_data["MonthlyCharges"] = monthly_charges

        if contract == "One year":
            if "Contract_One year" in input_data.columns:
                input_data["Contract_One year"] = 1

        elif contract == "Two year":
            if "Contract_Two year" in input_data.columns:
                input_data["Contract_Two year"] = 1

        probability = model.predict_proba(input_data)[0][1] * 100

        if probability < 30:
            risk = "LOW 🟢"
        elif probability < 70:
            risk = "MEDIUM 🟡"
        else:
            risk = "HIGH 🔴"

        st.subheader("Prediction Result")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                title={"text": "Churn Risk %"},
                gauge={
                    "axis": {"range": [0, 100]}
                }
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.metric(
            "Churn Probability",
            f"{probability:.2f}%"
        )

        st.write(f"## Risk Level: {risk}")

        if probability > 70:

            st.error("""
Recommended Actions:

• Offer discount

• Move customer to yearly contract

• Contact customer personally
""")

        elif probability > 30:

            st.warning("""
Recommended Actions:

• Loyalty rewards

• Promotional offer

• Customer engagement campaign
""")

        else:

            st.success("""
Customer likely to stay.

Maintain current relationship.
""")

# =========================
# MODEL PERFORMANCE PAGE
# =========================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.metric("Accuracy", "78.8%")

    st.divider()

    st.subheader("Classification Report")

    st.code("""
Precision (Churn): 62%

Recall (Churn): 52%

F1 Score (Churn): 56%

Overall Accuracy: 78.8%
""")

    st.subheader("Top Churn Drivers")

    churn_drivers = pd.DataFrame({
        "Feature": [
            "Fiber Optic Internet",
            "Streaming Movies",
            "Streaming TV",
            "Electronic Check",
            "Paperless Billing"
        ]
    })

    st.dataframe(
        churn_drivers,
        use_container_width=True
    )