import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# تحميل البيانات
file_path = '/Users/reemabalharith/Desktop/ai/FuelEconomy copy.csv'
data = pd.read_csv(file_path)

# تحويل العمود 'Extracurricular Activities' إلى أرقام
le = LabelEncoder()
data['Extracurricular Activities'] = le.fit_transform(data['Extracurricular Activities'])

# المدخلات والمخرجات
X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = data['Performance Index']

# تقسيم البيانات إلى مجموعات التدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = LinearRegression()
model.fit(X_train, y_train)

# واجهة Streamlit
st.title("Student Performance Prediction")
st.subheader("This project predicts student performance based on various factors")

# عرض DataFrame المستخدم
st.write("Below is the original DataFrame:")
st.write(data)

# عرض نموذج للتنبؤ
st.subheader("Enter the following details to predict the performance index:")

# مدخلات من المستخدم
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=6)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=80)
extracurricular_activities = st.selectbox("Extracurricular Activities", ['Yes', 'No'])
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=8)
sample_papers_practiced = st.number_input("Sample Papers Practiced", min_value=0, max_value=10, value=2)

# تحويل النشاطات اللامنهجية إلى رقم
extracurricular_activities_num = 1 if extracurricular_activities == 'Yes' else 0

# إجراء التنبؤ
input_data = [[hours_studied, previous_scores, extracurricular_activities_num, sleep_hours, sample_papers_practiced]]
predicted_performance = model.predict(input_data)

# عرض التنبؤ
st.subheader("Predicted Performance Index:")
st.write(f"The predicted performance index is: {predicted_performance[0]:.2f}")

# عرض خط بياني عشوائي كنموذج
st.subheader("Random Line Chart Example:")
chart_data = pd.DataFrame(np.random.randn(20, 4), columns=['p', 'q', 'r', 's'])
st.line_chart(chart_data)
