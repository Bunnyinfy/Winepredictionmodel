import streamlit as st
import numpy as np
import pickle

# Load the trained model (assuming the model is saved as 'wine_quality_model.pkl')
with open("wine_quality_model.pkl", "rb") as file:
    model = pickle.load(file)


# Define the prediction function
def predict_wine_quality(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return "Good Quality Wine" if prediction[0] == 1 else "Bad Quality Wine"


# Streamlit interface
st.title("Wine Quality Prediction")
st.title("Hey, Welcome to My page!")
st.write(
    """
This app uses a machine learning model to predict the quality of wine based on various chemical properties.
You can input the characteristics of the wine, and the model will predict whether it's a good quality wine or not.
"""
)

# Input fields for the features
fixed_acidity = st.number_input(
    "Fixed Acidity", min_value=0.0, max_value=20.0, value=7.5
)
volatile_acidity = st.number_input(
    "Volatile Acidity", min_value=0.0, max_value=2.0, value=0.5
)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.36)
residual_sugar = st.number_input(
    "Residual Sugar", min_value=0.0, max_value=20.0, value=6.1
)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=0.2, value=0.071)
free_sulfur_dioxide = st.number_input(
    "Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=17.0
)
total_sulfur_dioxide = st.number_input(
    "Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=102.0
)
density = st.number_input("Density", min_value=0.0, max_value=2.0, value=0.9978)
pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=3.35)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.8)
alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, value=10.5)

# Button to make the prediction
if st.button("Predict Wine Quality"):
    input_data = (
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
    )
    result = predict_wine_quality(input_data)
    st.success(result)
