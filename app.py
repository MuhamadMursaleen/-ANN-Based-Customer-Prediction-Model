# Import necessary libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained ANN model from the saved file (HDF5 format)
model = tf.keras.models.load_model('model.h5')

# Load the pre-trained label encoder for the 'Gender' column
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

# Load the pre-trained one-hot encoder for the 'Geography' column
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# Load the trained scaler to normalize input features
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Custom CSS for the color scheme
st.markdown("""
    <style>
    /* Set background color for the app */
    .stApp {
        background-color: #F1D3B2;
    }
    /* Style for the titles */
    .title {
        color: #46211A;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        color: #A43820;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    /* Styling the input section title */
    .sidebar-title {
        color: white;
        font-size: 22px;
        font-weight: bold;
    }
    /* Style input labels (Geography, Gender, etc.) */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #A43820 !important;
        font-weight: bold;
    }
    /* Footer style */
    .footer {
        color: grey;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app setup
st.markdown("<h1 class='title'>Customer Churn Prediction</h1>", unsafe_allow_html=True)  # App title
st.markdown("<h2 class='subtitle'>ANN Based Model</h2>", unsafe_allow_html=True)  # Subtitle

# Sidebar Input Section
st.sidebar.markdown("<h3 class='sidebar-title'>Input Customer Information</h3>", unsafe_allow_html=True)

# Collect user inputs with replaced color for input labels
geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0], key="geo")
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_, key="gender")
age = st.sidebar.slider('Age', 18, 92, key="age")
balance = st.sidebar.number_input('Balance', key="balance")
credit_score = st.sidebar.number_input('Credit Score', key="credit_score")
estimated_salary = st.sidebar.number_input('Estimated Salary', key="estimated_salary")
tenure = st.sidebar.slider('Tenure', 0, 10, key="tenure")
num_of_products = st.sidebar.slider('Number of Products', 1, 4, key="num_of_products")
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1], key="has_cr_card")
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1], key="is_active_member")

# Prepare input data by creating a DataFrame from the user inputs
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],  # Encode gender using the label encoder
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode the 'Geography' feature and convert it into a DataFrame
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()  # Encode geography using the one-hot encoder
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine the one-hot encoded 'Geography' column with the other input features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data using the pre-loaded scaler to match the model's training data format
input_data_scaled = scaler.transform(input_data)

# Predict the probability of customer churn using the trained ANN model
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]  # Extract the predicted probability

# Display the predicted probability of customer churn as a metric
st.markdown("<h3 style='color: #46211A;'>Prediction Results</h3>", unsafe_allow_html=True)
st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")

# Display a progress bar to represent the probability visually
progress = prediction_proba * 100
st.progress(int(progress))

# Provide a result interpretation based on the prediction threshold (0.5)
if prediction_proba > 0.5:
    st.markdown("<h4 class='prediction-output'>🚨 The customer is likely to churn.</h4>", unsafe_allow_html=True)
else:
    st.markdown("<h4 class='prediction-output'>✅ The customer is not likely to churn.</h4>", unsafe_allow_html=True)

# Add footer or summary information
st.markdown("<hr style='border: 2px solid #A43820;'>", unsafe_allow_html=True)
st.markdown("<p class='footer'>Built with TensorFlow and Streamlit</p>", unsafe_allow_html=True)
