import streamlit as st 
import pickle
import pandas as pd
import numpy as np

# Load the dataset
dataset = pd.read_csv(r'C:\Users\Thite\OneDrive\Desktop\Bengaluru-house-price-prediction\bengaluru_house_dataset.csv')



def load_model():
    with open('model.pkl', 'rb') as file:
        model, columns = pickle.load(file)
    return model, columns

model, columns = load_model()

st.title("Bangalore House price prediction")
# Extract unique locations
unique_locations = dataset['location'].unique()

# Prediction function
def predict_price(location, sqft, bath, bhk):    
    loc_index = np.where(columns == location)[0][0]

    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

selected_location = st.selectbox("Select Location", unique_locations)
filtered_data = dataset[dataset['location'] == selected_location]

total_sqft = st.number_input("Total Square Feet", min_value=0, value=10000)
bath = st.number_input("Number of Bathrooms", min_value=1, value=20)
bhk = st.number_input("Number of BHK", min_value=1, value=10)

if st.button("Predict"):
    prediction = predict_price(selected_location, total_sqft, bath, bhk)
    st.write(f"Predicted Price for {selected_location}: {prediction *100000}")