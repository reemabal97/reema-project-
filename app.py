from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# إنشاء التطبيق
app = Flask(__name__)

# تحميل البيانات وتدريب النموذج
file_path = '/Users/reemabalharith/Desktop/Student_Performance.csv'
data = pd.read_csv(file_path)

# تحويل العمود 'Extracurricular Activities' إلى أرقام
le = LabelEncoder()
data['Extracurricular Activities'] = le.fit_transform(data['Extracurricular Activities'])

# المدخلات والمخرجات
X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = data['Performance Index']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = LinearRegression()
model.fit(X_train, y_train)

# حفظ النموذج
joblib.dump(model, 'model.pkl')

# تحميل النموذج
model = joblib.load('model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # الحصول على المدخلات من المستخدم
        hours_studied = float(request.form['hours_studied'])
        previous_scores = float(request.form['previous_scores'])
        extracurricular_activities = 1 if request.form['extracurricular_activities'].lower() == 'yes' else 0
        sleep_hours = float(request.form['sleep_hours'])
        sample_papers_practiced = float(request.form['sample_papers_practiced'])

        # التنبؤ باستخدام النموذج
        prediction = model.predict([[hours_studied, previous_scores, extracurricular_activities, sleep_hours, sample_papers_practiced]])

        # إرجاع النتيجة إلى الصفحة
        return render_template('index.html', prediction_text=f'Predicted Performance Index: {prediction[0]:.2f}')

    except ValueError:
        return render_template('index.html', prediction_text="Please enter valid numbers.")


if __name__ == "__main__":
    app.run(debug=True)
