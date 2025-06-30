import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Customer Churn Prediction')
st.write("Enter the customer's features to predict churn.")
try:
    lgbm_model = joblib.load('lgbm_churn_model.joblib')
except FileNotFoundError:
    st.error("Error: LightGBM model file 'lgbm_churn_model (1).joblib' not found.")
    st.stop()
try:
    scaler = joblib.load('scaler (1).joblib')
except FileNotFoundError:
    st.error("Error: Scaler file 'scaler.joblib' not found.")
    st.stop()
if hasattr(lgbm_model, 'feature_name_'):
    model_feature_columns = list(lgbm_model.feature_name_)
elif hasattr(lgbm_model, 'feature_name'):
     model_feature_columns = list(lgbm_model.feature_name)
else:
    st.error("Error: Could not retrieve feature names from the trained model.")
    st.stop()
categorical_cols_info = {
    'X18': ['1-3 years', '6-12 months'], 
    'X151': ['CA', 'CO', 'DC', 'FL', 'MA', 'MD', 'NJ', 'NY', 'OR', 'PA', 'TN', 'TX', 'VA', 'WA'], 
    'X155': ['A', 'B'], 
    'X156': ['Garden', 'High Rise', 'Podium'], 
    'X157': ['suburban', 'urban'] 
}
numerical_cols_for_scaling = list(scaler.feature_names_in_)
input_data = {}
st.sidebar.header('Feature Input')
st.sidebar.subheader('Categorical Features')
for col, values in categorical_cols_info.items():
    input_data[col] = st.sidebar.selectbox(f'Select value for {col}', options=values)
st.sidebar.subheader('Numerical Features')
for col in numerical_cols_for_scaling:
     input_data[col] = st.sidebar.number_input(f'Enter value for {col}', value=0.0)

if st.button('Predict Churn'):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=categorical_cols_info.keys(), dummy_na=False)
    for col in model_feature_columns:
        if col not in input_df.columns:
            if col in numerical_cols_for_scaling:
                 st.warning(f"Numerical column '{col}' not found in input. Filling with median from training data.")
                 input_df[col] = 0 
            else:
                input_df[col] = 0 
    input_df = input_df[model_feature_columns]
    cols_to_scale_in_input = [col for col in numerical_cols_for_scaling if col in input_df.columns]
    if cols_to_scale_in_input:
        input_df[cols_to_scale_in_input] = scaler.transform(input_df[cols_to_scale_in_input])
    else:
        st.warning("No numerical columns found in input to apply scaling.")
    prediction = lgbm_model.predict(input_df)
    prediction_proba = lgbm_model.predict_proba(input_df)[:, 1]

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error(f"Prediction: Customer is likely to Churn (Probability: {prediction_proba[0]:.4f})")
    else:
        st.success(f"Prediction: Customer is unlikely to Churn (Probability: {prediction_proba[0]:.4f})")