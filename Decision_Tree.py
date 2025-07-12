import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = "F:\\AWFERA\\Machine learning\\AwferaMachineLearningProjects\\diabetes.csv"
df = pd.read_csv(file_path)

# Step 2: Handle Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)
for col in df.select_dtypes(include=['object']):
    df[col].fillna(df[col].mode()[0], inplace=True)

# Step 3: Prepare Data
x = df.drop(columns=['Outcome'])
y = df['Outcome']

# Step 4: Standard Scaling
scler = StandardScaler()
x_scaled = scler.fit_transform(x)

# Step 5: Split Data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

# Step 6: Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# ---------------------- Streamlit Web App ----------------------
st.title("🧠 Diabetes Prediction App")
st.markdown("Enter patient data in the sidebar to check if they are diabetic.")

# Sidebar Input
st.sidebar.header("Enter Patient Information:")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=120)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=85)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)

if st.sidebar.button("Predict"):
    # Step 7: Predict New Data
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]])
    user_data_scaled = scler.transform(user_data)
    prediction = model.predict(user_data_scaled)[0]

    # Display Result
    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.error("🔴 The person is **Diabetic**")
    else:
        st.success("🟢 The person is **Not Diabetic**")

    # Step 8: Model Accuracy
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"✅ Model Accuracy: **{acc:.2f}**")

    # Step 9: Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    fig1, ax1 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
                cmap='Blues', xticklabels=["Non-Diabetic", "Diabetic"],
                yticklabels=["Non-Diabetic", "Diabetic"], ax=ax1)
    st.pyplot(fig1)
