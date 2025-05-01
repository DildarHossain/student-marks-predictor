import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Load and clean data
df = pd.read_csv("student_info.csv")
df = df.dropna()

# Train the model
X = df[['study_hours']]
y = df['student_marks']
model = LinearRegression()
model.fit(X, y)

# Predict and classify
df['predicted_marks'] = model.predict(X)
df['result'] = df['predicted_marks'].apply(lambda x: 'Pass' if x >= 60 else 'Fail')


# Streamlit App UI
st.title("📘 Student Marks Predictor")
st.write("Enter the number of study hours:")

hours = st.number_input("Study Hours", min_value=0.0, step=0.5)

if st.button("Predict Marks"):
    predicted = model.predict([[hours]])[0]
    result = "Pass" if predicted >= 60 else "Fail"

    st.success(f"Predicted Marks: {predicted:.2f} → **{result}**")

st.subheader("📊 Full Data with Predictions")
st.dataframe(df)
