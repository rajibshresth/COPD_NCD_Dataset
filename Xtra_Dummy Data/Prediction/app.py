import streamlit as st
import pandas as pd
import pickle

# Load the trained models
with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Define the expected feature names
feature_names = [
    'Age',
    'Biomass Fuel',
    'Tobacco Smoking',
    'Outdoor Air Poll',
    'Occupational Expos',
    'Family History',
    'Respiratory Infection',
    'Age_Smoking_Interaction',
    'Income_Biomass_Interaction',
    'Location_Urban',
    'Education Level_Primary',
    'Education Level_Secondary',
    'Occupation_Farmer',
    'Occupation_Office',
    'Occupation_Other',
    'Age Group_50-59',
    'Age Group_60-69',
    'Age Group_70-79',
    'Gender_Male',
    'Income Level_Low',
    'Income Level_Middle',
    'Health Insurance_Yes',
]

# Function to make predictions
def predict_copd(features, model):
    return model.predict(features)

# Streamlit app layout
st.title("COPD Diagnosis Predictor")

# User inputs
st.header("Input Features")
age = st.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", options=["Male", "Female"])
smoking_status = st.selectbox("Smoking Status", options=["Non-Smoker", "Ex-Smoker", "Current Smoker"])
biomass_fuel = st.selectbox("Biomass Fuel", options=["No", "Yes"])
income_level = st.selectbox("Income Level", options=["Low", "Middle", "High"])
education_level = st.selectbox("Education Level", options=["None", "Primary", "Secondary"])
family_history = st.selectbox("Family History", options=["No", "Yes"])
health_insurance = st.selectbox("Health Insurance", options=["No", "Yes"])
occupational_expos = st.selectbox("Occupational Exposures", options=["No", "Yes"])
outdoor_air_poll = st.selectbox("Outdoor Air Pollution", options=["No", "Yes"])
respiratory_infection = st.selectbox("Respiratory Infection History", options=["No", "Yes"])
tobacco_smoking = st.selectbox("Tobacco Smoking", options=["No", "Yes"])

# Prepare input data as a DataFrame
input_data = {
    'Age': [age],
    'Biomass Fuel': [1 if biomass_fuel == "Yes" else 0],
    'Tobacco Smoking': [1 if smoking_status == "Current Smoker" else 0],
    'Outdoor Air Poll': [1 if outdoor_air_poll == "Yes" else 0],
    'Occupational Expos': [1 if occupational_expos == "Yes" else 0],
    'Family History': [1 if family_history == "Yes" else 0],
    'Respiratory Infection': [1 if respiratory_infection == "Yes" else 0],
    'Age_Smoking_Interaction': [age * (1 if smoking_status == "Current Smoker" else 0)],
    'Income_Biomass_Interaction': [0],
    'Location_Urban': [0],
    'Education Level_Primary': [1 if education_level == "Primary" else 0],
    'Education Level_Secondary': [1 if education_level == "Secondary" else 0],
    'Occupation_Farmer': [0],
    'Occupation_Office': [0],
    'Occupation_Other': [0],
    'Age Group_50-59': [1 if 50 <= age < 60 else 0],
    'Age Group_60-69': [1 if 60 <= age < 70 else 0],
    'Age Group_70-79': [1 if 70 <= age < 80 else 0],
    'Gender_Male': [1 if gender == "Male" else 0],
    'Income Level_Low': [1 if income_level == "Low" else 0],
    'Income Level_Middle': [1 if income_level == "Middle" else 0],
    'Health Insurance_Yes': [1 if health_insurance == "Yes" else 0],
}

# Create DataFrame with expected feature columns
features = pd.DataFrame(input_data)

# Align with training feature names
features_encoded = features.reindex(columns=feature_names, fill_value=0)

# Make predictions
if st.button("Predict"):
    lr_prediction = predict_copd(features_encoded, lr_model)
    rf_prediction = predict_copd(features_encoded, rf_model)

    st.write(f"Logistic Regression Prediction: {'Undiagnosed' if lr_prediction[0] == 1 else 'Diagnosed'}")
    st.write(f"Random Forest Prediction: {'Undiagnosed' if rf_prediction[0] == 1 else 'Diagnosed'}")
