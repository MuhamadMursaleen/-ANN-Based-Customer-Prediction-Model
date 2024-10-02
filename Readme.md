Here’s a combined README file that integrates both your previous content and the new structure for your project. This version maintains clarity and is beginner-friendly, guiding users through the setup and usage of your Customer Churn Prediction project.

---

# ANN-Based Customer Churn Prediction Model

This project demonstrates the creation of an Artificial Neural Network (ANN) model for predicting customer behavior using a dataset of 10,000 bank customers. The dataset contains various features related to the customers' demographics and financial information. The goal is to predict a target outcome (e.g., churn or retention) based on these features.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Model Architecture](#model-architecture)
- [Installation and Setup](#installation-and-setup)
- [Running the Project](#running-the-project)
- [Future Work](#future-work)
- [License](#license)

## Project Overview

This project utilizes an Artificial Neural Network (ANN) to analyze a bank's customer dataset and make predictions about customer behavior. The ANN was built using the TensorFlow/Keras library and trained on features such as credit score, age, account balance, and other customer-related attributes.

## Dataset Information

The dataset contains information on 10,000 customers of a bank with the following features:

- **CreditScore**: The customer’s credit score.
- **Geography**: The country where the customer is based.
- **Gender**: Male or Female.
- **Age**: The customer’s age.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: The customer's bank balance.
- **NumOfProducts**: Number of products the customer has with the bank.
- **HasCrCard**: Does the customer have a credit card? (0 = No, 1 = Yes)
- **IsActiveMember**: Is the customer an active member? (0 = No, 1 = Yes)
- **EstimatedSalary**: The customer’s estimated salary.

## Model Architecture

The ANN model was built using the following architecture:

- **Input Layer**: 10 input features.
- **Hidden Layers**:
  - Fully connected layers with activation functions (e.g., ReLU).
  - Dropout layers to prevent overfitting.
- **Output Layer**: Binary output for prediction (or softmax for multi-class predictions).

The model was trained using the Adam optimizer and binary cross-entropy loss for binary classification problems.

## Installation and Setup

Follow these steps to set up the project on your local machine.

### Step 1: Clone the Repository

Open your terminal and clone the repository using:

```bash
git clone https://github.com/MuhamadMursaleen/-ANN-Based-Customer-Prediction-Model.git
cd -ANN-Based-Customer-Prediction-Model
```

### Step 2: Create a Virtual Environment

It's recommended to create a virtual environment to manage dependencies. Run the following command:

```bash
# For Python 3
python3 -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

Once the environment is activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

### Step 4: Verify the Installation

Ensure that all required packages are installed by running:

```bash
python --version
pip freeze
```

## Running the Project

### Step 1: Prepare the Dataset

Place the `churn_modelling.csv` dataset file in the root directory of the project. This dataset is used to train the ANN model.

### Step 2: Train the Model

To train the ANN model using the provided dataset, you can use either the Python script or the Jupyter Notebook:

- **Using Python Script**:

```bash
python train_model.py
```

- **Using Jupyter Notebook**:

Open `train_model.ipynb` in Jupyter Notebook and run all cells.

### Step 3: Make Predictions

Once the model is trained, you can use it to make predictions on new customer data:

- **Using Python Script**:

Prepare your input data in `customer_data.csv`, then run:

```bash
python predict.py --input customer_data.csv
```

- **Using Jupyter Notebook**:

Open `predict.ipynb`, load your `customer_data.csv`, and run all cells to see predictions.

### Step 4: Run the Streamlit App

For a user-friendly interface, you can run the Streamlit app:

```bash
streamlit run app.py
```

This will open a new tab in your web browser where you can enter customer data and see the predicted churn probability.

## Future Work

- Enhance the model's architecture to improve accuracy.
- Experiment with other algorithms like Random Forest or XGBoost for comparison.
- Implement hyperparameter tuning.
- Add more detailed feature engineering.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

