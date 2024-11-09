import streamlit as st
import pandas as pd
import numpy as np
import pickle
import six
import joblib
import sys
from datetime import datetime
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

# Load model (replace 'model.sav' with the actual model filename)
model = joblib.load('model.sav')

def predict(model, input_df):
    # Use model.predict to make predictions
    predictions = model.predict(input_df)
    return predictions[0]  # Assuming you want the first prediction for single input

def run():
    # Load and resize images
    image = Image.open('ChurnPHeading.png')
    image = image.resize((300, 100))
    image_meeting_room = Image.open('ChurnP.png')
    image_meeting_room = image_meeting_room.resize((300, 100))
    
    # Display images and sidebar options
    st.sidebar.image(image)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch", "About")
    )
    st.sidebar.info('This app is created to predict Churn rate')
    st.sidebar.image(image_meeting_room)
    st.title("Telecom Churn Prediction")
    
    # Logic for the "Online" selection
    if add_selectbox == "Online":
        user_gender = st.selectbox("Select Gender:", ["Male", "Female"])
        user_phoneServices = st.selectbox("Select Phone Services:", ["Yes", "No"])
        user_InternetServices = st.selectbox("Use Internet Services:", ["Yes", "No"])
        if user_InternetServices == "Yes":
            user_KindOFservice = st.selectbox("Select Internet Services:", ["Fiber Optic", "DSL"])
        else:
            user_KindOFservice=0
        user_StreamingTV = st.selectbox("TV Streaming:", ["Yes", "No"])
        user_StreamingMovie = st.selectbox("Movie Streaming:", ["Yes", "No"])
        user_ContractType = st.selectbox("Contract Type:", ["Monthly", "Yearly", "Two Year"])
        user_PaymentMethod = st.selectbox("Payment Method:", ["Bank Transfer", "Credit Card", "Electronic Check", "Mailed Check"])
        user_tenureGroup = st.slider('Tenure Group', min_value=1, max_value=72, step=12, value=1)
        user_monthlyCharges = st.number_input("Enter Monthly Charges:", value=0.0, format="%.2f")
        user_TotalCharges = st.number_input('Total Charges', value=0.0, format="%.2f")
        
        # Create input dictionary
        input_data = {
            "MonthlyCharges": [user_monthlyCharges],
            "TotalCharges": [user_TotalCharges],
            "Gender": [0 if user_gender == "Male" else 1],
            "PhoneService": [1 if user_phoneServices == "Yes" else 0],
            "InternetService_No": [1 if user_InternetServices == "Yes" else 0],
            "InternetService_Fiber optic": [1 if user_KindOFservice == "Fiber Optic" else 0],
            "InternetService_DSL": [2 if user_KindOFservice == "DSL" else 1],
            "StreamingTV": [1 if user_StreamingTV == "Yes" else 0],
            "StreamingMovies": [1 if user_StreamingMovie == "Yes" else 0],
            "Contract_Month-to-month": [1 if user_ContractType == "Monthly" else 0],
            "Contract_One year": [1 if user_ContractType == "Yearly" else 0],
            "Contract_Two year": [1 if user_ContractType == "Two Year" else 0],
            "PaymentMethod_Bank transfer (automatic)": [1 if user_PaymentMethod == "Bank Transfer" else 0],
            "PaymentMethod_Credit card (automatic)": [1 if user_PaymentMethod == "Credit Card" else 0],
            "PaymentMethod_Electronic check": [1 if user_PaymentMethod == "Electronic Check" else 0],
            "PaymentMethod_Mailed check": [1 if user_PaymentMethod == "Mailed Check" else 0],
            "tenure_group_1 - 12": [1 if 1 <= user_tenureGroup <= 12 else 0],
            "tenure_group_13 - 24": [1 if 13 <= user_tenureGroup <= 24 else 0],
            "tenure_group_25 - 36": [1 if 25 <= user_tenureGroup <= 36 else 0],
            "tenure_group_37 - 48": [1 if 37 <= user_tenureGroup <= 48 else 0],
            "tenure_group_49 - 60": [1 if 49 <= user_tenureGroup <= 60 else 0],
            "tenure_group_61 - 72": [1 if 61 <= user_tenureGroup <= 72 else 0]    
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Predict when the button is clicked
        if st.button("Predict"):
            prediction = predict(model, input_df)
            prediction_label = "Churn" if prediction == 1 else "Not Churn"
            st.success(f'The prediction is: {prediction_label}')

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])
        
        if file_upload is not None:
            # Load data
            data = pd.read_csv(file_upload)
            
            # Remove unwanted columns like 'Unnamed: 0'
            data = data.drop(columns=['Unnamed: 0'], errors='ignore')
            
            # Define expected columns
            expected_columns = ['MonthlyCharges', 'TotalCharges', 'Gender', 
                                'PhoneService', 'InternetService_No', 
                                'InternetService_Fiber optic', 'InternetService_DSL', 
                                'StreamingTV', 'StreamingMovies', 
                                'Contract_Month-to-month', 'Contract_One year', 
                                'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', 
                                'PaymentMethod_Credit card (automatic)', 
                                'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 
                                'tenure_group_1 - 12', 'tenure_group_13 - 24', 
                                'tenure_group_25 - 36', 'tenure_group_37 - 48', 
                                'tenure_group_49 - 60', 'tenure_group_61 - 72']
            
            # Select only the columns needed for prediction
            data = data[expected_columns]
            
            # Make predictions
            predictions = model.predict(data)
            
            # Create an output DataFrame
            output = pd.DataFrame({
                'Customer_id': data.index if 'Customer_id' not in data.columns else data['Customer_id'],
                'Churn': predictions
            })
    
            # Allow download of the predictions as a CSV file
            csv = output.to_csv(index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )

    if add_selectbox == 'About':
        st.subheader("Built with Streamlit and Pycaret")
        st.subheader("Nishtha Jain")
        # st.subheader("https://www.linkedin.com/in/hunaidkhan/")
        # st.button("Re-run")

    
# Run the function
run()
