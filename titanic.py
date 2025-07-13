import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("titanic_dataset.csv")

# Preprocessing
age_imputer = SimpleImputer(strategy='median')
data['age'] = age_imputer.fit_transform(data[['age']])
data['embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(data[['embarked']]).ravel()

data.drop(columns=['deck'], inplace=True)

le_sex = LabelEncoder()
le_embarked = LabelEncoder()
data['sex'] = le_sex.fit_transform(data['sex'])
data['embarked'] = le_embarked.fit_transform(data['embarked'])

data.drop(columns=['class', 'who', 'adult_male', 'embark_town', 'alive', 'alone'], inplace=True)

# Train model
x = data.drop('survived', axis=1)
y = data['survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Streamlit UI
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details below to predict survival.")

# User Inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Passenger Fare", 0.0, 600.0, 30.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Preprocess user input
sex_encoded = le_sex.transform([sex])[0]
embarked_encoded = le_embarked.transform([embarked])[0]
user_input = [[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]]

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(user_input)
    result = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
    st.success(f"Prediction: {result}")
