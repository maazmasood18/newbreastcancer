import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def load_data():
    data = pd.read_csv('gbsg.csv')  # Make sure your dataset is in the correct location in the repo
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data.drop(['pid', 'rfstime'], axis=1, inplace=True)
    return data

# Train the Random Forest model
def train_model(data):
    features = ['age', 'meno', 'size', 'grade', 'nodes', 'pgr', 'er', 'hormon']
    X = data[features]
    y = data['status']
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train RandomForest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=52)
    rf_classifier.fit(X_train, y_train)
    
    return rf_classifier, scaler

# Prediction function
def predict_status(model, scaler, user_input):
    # Scale the input
    user_input_scaled = scaler.transform([user_input])
    # Make prediction
    prediction = model.predict(user_input_scaled)
    return prediction

# Streamlit app layout
st.title('Interactive Breast Cancer Prediction App')

st.write("This app allows you to input medical details and predicts whether the patient's status is 0 (alive without recurrence) or 1 (recurrence or death).")

# Load data and train model
data = load_data()
model, scaler = train_model(data)

# Collect user inputs
st.header('Enter Patient Details')
age = st.number_input('Age', min_value=20, max_value=100, value=50)
meno = st.selectbox('Menopausal Status (1 = pre-menopausal, 2 = post-menopausal)', [1, 2])
size = st.number_input('Tumor Size (mm)', min_value=0, max_value=150, value=30)
grade = st.selectbox('Tumor Grade (1-3)', [1, 2, 3])
nodes = st.number_input('Number of Positive Nodes', min_value=0, max_value=50, value=1)
pgr = st.selectbox('Progesterone Receptor (0 = negative, 1 = positive)', [0, 1])
er = st.selectbox('Estrogen Receptor (0 = negative, 1 = positive)', [0, 1])
hormon = st.selectbox('Hormonal Therapy (0 = no, 1 = yes)', [0, 1])

# Create an input array
user_input = [age, meno, size, grade, nodes, pgr, er, hormon]

# Prediction button
if st.button('Predict Status'):
    prediction = predict_status(model, scaler, user_input)
    if prediction == 0:
        st.success('Prediction: Alive without recurrence (Status 0)')
    else:
        st.error('Prediction: Recurrence or death (Status 1)')

# Optionally display the dataset and visualizations
if st.checkbox('Show Dataset'):
    st.write(data)

# You can still add the data visualizations if needed
