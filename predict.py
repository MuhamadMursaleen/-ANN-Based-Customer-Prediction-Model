import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import argparse

# Load the trained model and encoders/scaler
def load_resources():
    """Load the trained model and preprocessing resources."""
    model = tf.keras.models.load_model('model.h5')

    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    return model, label_encoder_gender, onehot_encoder_geo, scaler

# Preprocess the input data
def preprocess_input_data(input_file, label_encoder_gender, onehot_encoder_geo, scaler):
    """Preprocess input data for prediction."""
    # Load input data
    data = pd.read_csv(input_file)

    # Handle missing values
    data['Geography'].fillna('Unknown', inplace=True)
    data['Gender'].fillna('Unknown', inplace=True)

    # Encode 'Gender' column
    data['Gender'] = label_encoder_gender.transform(data['Gender'])

    # One-hot encode 'Geography' column
    geo_encoded = onehot_encoder_geo.transform(data[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with the original data
    data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)

    # Ensure all expected columns are present
    expected_columns = scaler.feature_names_in_
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0  # Add missing columns with default value 0

    # Reorder columns to match the order expected by the scaler
    data = data[expected_columns]

    # Scale the features
    X_scaled = scaler.transform(data)

    return X_scaled

# Make predictions
def make_predictions(model, input_file):
    """Make predictions on the input data."""
    _, label_encoder_gender, onehot_encoder_geo, scaler = load_resources()
    X_scaled = preprocess_input_data(input_file, label_encoder_gender, onehot_encoder_geo, scaler)
    
    # Predict churn probabilities
    predictions = model.predict(X_scaled)
    return predictions

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict customer churn.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file with customer data.')
    
    args = parser.parse_args()
    
    # Load the model first
    model, _, _, _ = load_resources()  # Only unpacking the model here
    
    predictions = make_predictions(model, args.input)

    # Output predictions
    for i, prob in enumerate(predictions):
        print(f'Customer {i+1}: Churn Probability: {prob[0]:.2f}')
