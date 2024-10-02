# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
import pickle
import datetime

# Load and preprocess the dataset
def load_data(file_path):
    """Load dataset from a CSV file and return the data."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the data by encoding categorical variables and scaling."""
    
    # Drop columns that are not needed for prediction
    data = data.drop(['RowNumber','CustomerId', 'Surname'], axis=1)
    
    # Encode the 'Gender' column
    label_encoder_gender = LabelEncoder()
    data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
    
    # One-hot encode the 'Geography' column
    onehot_encoder_geo = OneHotEncoder()
    geo_encoded = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    
    # Combine one-hot encoded columns with the original data
    data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)
    
    # Features and target variable
    X = data.drop('Exited', axis=1)  # Ensure 'Exited' is the target variable
    y = data['Exited']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save encoders and scaler for later use in predictions
    with open('label_encoder_gender.pkl', 'wb') as file:
        pickle.dump(label_encoder_gender, file)
    
    with open('onehot_encoder_geo.pkl', 'wb') as file:
        pickle.dump(onehot_encoder_geo, file)
    
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    
    return X_scaled, y

# Build the ANN model
def build_model(input_shape):
    """Build and compile a Sequential ANN model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_dim=input_shape, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(X_train, y_train, X_test, y_test):
    """Train the ANN model with the given training data."""
    
    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)
    # Build the model
    model = build_model(X_train.shape[1])
    
    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, 
                        callbacks=[tensorflow_callback, early_stopping])
    
    # Save the final model
    model.save('model.h5')
    return model, history

# Main function to execute the training process
if __name__ == "__main__":
    # Load the dataset
    data_file_path = 'Churn_Modelling.csv'  # Updated file path
    data = load_data(data_file_path)
    
    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model, history = train_model(X_train, y_train, X_test, y_test)

    # Output a message once training is complete
    print("Model training complete. Model saved as 'model.h5'.")
