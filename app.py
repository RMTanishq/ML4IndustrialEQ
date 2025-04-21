import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load pre-trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Define feature names
feature_names = [
    'operation_cycles', 'vibration', 'error_count', 'motor_temp',
    'lubricant_level', 'load', 'warning_flag', 'recent_repairs', 'motor_current'
]

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("🔧 Predictive Maintenance for Industrial Equipment")

# Sidebar user input
st.sidebar.header("Enter Sensor Readings")

user_input = {}
for feature in feature_names:
    if feature == "warning_flag":
        user_input[feature] = st.sidebar.selectbox(feature, [0, 1])
    else:
        user_input[feature] = st.sidebar.number_input(feature, value=0.0)

input_df = pd.DataFrame([user_input])
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
probability = model.predict_proba(scaled_input)[0][1]

# Show input as a table
st.subheader("📋 Sensor Readings")
st.dataframe(input_df)

# Display result
st.subheader("🧠 Model Prediction")
if prediction == 1:
    st.error(f"⚠️ Maintenance Required (Probability: {probability:.2f})")
else:
    st.success(f"✅ Operating Normally (Probability: {1 - probability:.2f})")

# Probability Bar
st.subheader("🔍 Prediction Probability")
st.progress(probability)

# Feature Importance Plot
st.subheader("📊 Feature Importance")
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=importances[sorted_indices], y=np.array(feature_names)[sorted_indices], ax=ax)
ax.set_title("Top Influential Sensors")
st.pyplot(fig)
