import streamlit as st
import pandas as pd
import pickle

# Load the trained models and get feature names from one model
models = {}
model_names = ['Random_Forest', 'Linear_Regression', 'Support_Vector_Regression', 'Gradient_Boosting']
for name in model_names:
    with open(f'{name}.pkl', 'rb') as f:
        models[name] = pickle.load(f)

# Assume all models have the same features and use one model to get feature names
sample_model = models['Random_Forest']
feature_names = sample_model.feature_names_in_

# Streamlit app layout
st.title("Noncommunicable Diseases Prediction in Nepal")

# Input features
st.sidebar.header("Input Features")
disease_risk_factors = st.sidebar.selectbox("Diseases & Risk Factors", ["Noncommunicable diseases", "Air pollution", "Harmful Alcohol Use",
                                                                        "Cancer", "Chronic respiratory diseases", "Cardiovascular diseases", 
                                                                        "Diabetes", "Obesity/unhealthy diet", "Physical inactivity", "Tobacco Use"])
gender = st.sidebar.selectbox("Gender", ["Males", "Females"])
year_bin = st.sidebar.selectbox("Year Bin", ['1980s', '1990s', '2000s', '2010s'])

# Create a feature DataFrame with all the necessary columns
input_data = pd.DataFrame(columns=feature_names)

# Initialize with zeros
input_data.loc[0] = [0] * len(feature_names)

# Set appropriate values based on user input
input_data.at[0, 'Diseases & Risk Factors_Noncommunicable diseases'] = 1
input_data.at[0, 'Gender_Males'] = 1 if gender == "Males" else 0
input_data.at[0, 'Year_Bin_2000s'] = 1 if year_bin == "2000s" else 0
input_data.at[0, 'Year_Bin_2010s'] = 1 if year_bin == "2010s" else 0

# Show input data for debugging
st.write("Input Data:")
st.write(input_data)

# Prediction
if st.button("Predict"):
    predictions = {}
    for name, model in models.items():
        prediction = model.predict(input_data)[0]
        predictions[name] = prediction

    st.write("Predictions:")
    for name, pred in predictions.items():
        st.write(f"{name}: {pred:.2f}")

