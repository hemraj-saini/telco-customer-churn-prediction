import pandas as pd
import joblib
import streamlit as st

# Load Model

model=joblib.load('Telco_Customer_Churn.joblib')

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction")

# User Inputs

gender = st.selectbox(
    "Customer Gender",
    ["Male", "Female"]
)

SeniorCitizen = st.selectbox(
    "Is the customer a senior citizen (60+ years)?",
    [0, 1]
)

Partner = st.selectbox(
    "Does the customer have a partner?",
    ["Yes", "No"]
)

Dependents = st.selectbox(
    "Does the customer have dependents?",
    ["Yes", "No"]
)

tenure = st.number_input(
    "How many months has the customer stayed with the company?",
    min_value=0
)

PhoneService = st.selectbox(
    "Does the customer have phone service?",
    ["Yes", "No"]
)

MultipleLines = st.selectbox(
    "Does the customer use multiple phone lines?",
    ["Yes", "No"]
)

InternetService = st.selectbox(
    "Internet Service Type",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Has Online Security Service?",
    ["Yes", "No"]
)

OnlineBackup = st.selectbox(
    "Has Online Backup Service?",
    ["Yes", "No"]
)

DeviceProtection = st.selectbox(
    "Has Device Protection Service?",
    ["Yes", "No"]
)

TechSupport = st.selectbox(
    "Has Technical Support Service?",
    ["Yes", "No"]
)

StreamingTV = st.selectbox(
    "Uses Streaming TV?",
    ["Yes", "No"]
)

StreamingMovies = st.selectbox(
    "Uses Streaming Movies?",
    ["Yes", "No"]
)

Contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Uses Paperless Billing?",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Bill Amount ($)",
    min_value=0.0
)

TotalCharges = st.number_input(
    "Total Amount Paid So Far ($)",
    min_value=0.0
)

if st.button("Predict Churn"):

    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer Likely To Churn")
    else:
        st.success("✅ Customer Likely To Stay")

    st.write(
        f"Churn Probability: {probability:.2%}"
    )