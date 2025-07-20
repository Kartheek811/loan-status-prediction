import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# Load and preprocess the training data
@st.cache_resource
def load_model():
    df = pd.read_csv("data/train.csv")
    df = df.dropna()

    le = LabelEncoder()
    for column in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
        df[column] = le.fit_transform(df[column])

    X = df[['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome', 'LoanAmount', 'Credit_History', 'Property_Area']]
    y = df['Loan_Status']

    model = SVC(kernel='linear')
    model.fit(X, y)
    return model, le

model, le = load_model()

st.title("🏦 Loan Status Prediction")

st.write("Enter the applicant details below:")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", value=5000)
loan_amount = st.number_input("Loan Amount", value=100)
credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

if st.button("Predict Loan Status"):
    # Manually encode input values
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0
    credit_val = 1 if credit_history == "Good (1)" else 0
    property_map = {"Urban": 2, "Rural": 1, "Semiurban": 0}
    property_val = property_map[property_area]

    input_data = pd.DataFrame([[
        gender_val, married_val, education_val, self_employed_val,
        applicant_income, loan_amount, credit_val, property_val
    ]], columns=[
        'Gender', 'Married', 'Education', 'Self_Employed',
        'ApplicantIncome', 'LoanAmount', 'Credit_History', 'Property_Area'
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
