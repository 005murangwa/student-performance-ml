🎓 Student Performance Predictor (Django + Machine Learning)

A Django-based Machine Learning web application that predicts a student’s academic performance score based on daily study habits, sleep, prior exam results, and extracurricular participation.

The system uses a trained ML regression model served through a Django API and connected to a clean, interactive frontend.

🚀 Features

📊 Predicts student performance score (0–100)

🧠 Machine Learning model (trained with scikit-learn)

🌐 Django REST API (/api/predict/)

🖥️ User-friendly web interface

✅ Input validation for realistic values

💾 Pre-trained model loaded from .pkl file

🔁 Reproducible environment using requirements.txt

🧩 Inputs Used for Prediction
Feature	Description
Hours studied per day	Average daily study hours
Sleep hours per night	Average sleep duration
Previous exam score	Last exam score (0–100)
Sample papers practiced	Number of practice papers
Extracurricular activities	Yes / No
📈 Output

Predicted Performance Score (out of 100)

Performance category (e.g. Normal / Balanced)

Personalized academic advice

🗂️ Project Structure
student_ml/
│
├── student_ml/              # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── performance/             # Main ML app
│   ├── migrations/
│   ├── model/
│   │   └── model.pkl        # Trained ML model
│   ├── views.py             # API + frontend logic
│   ├── urls.py
│   └── serializers.py
│
├── templates/               # HTML templates
│   └── index.html
│
├── static/                  # CSS, JS, images
│
├── manage.py
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Technologies Used

Backend: Django, Django REST Framework

ML: scikit-learn, pandas, joblib

Frontend: HTML, CSS, JavaScript

Environment: Python 3.12 + Virtual Environment

🔧 Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/005murangwa/student-performance-ml.git
cd student-performance-predictor

2️⃣ Create and Activate Virtual Environment
py -3.12 -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Migrations
python manage.py migrate

5️⃣ Start the Development Server
python manage.py runserver


Open your browser at:

http://127.0.0.1:8000/

🔌 API Endpoint
POST /api/predict/
Example JSON Request:
{
  "hours_studied": 8,
  "sleep_hours": 7,
  "previous_score": 80,
  "sample_papers": 15,
  "extracurricular": true
}

Example Response:
{
  "predicted_score": 80.92,
  "category": "Normal / Balanced",
  "message": "You're in a strong zone—keep consistency and avoid last-minute cramming."
}

🧠 Machine Learning Model

Algorithm: Linear Regression

Trained using scikit-learn

Stored as: performance/model/model.pkl

Loaded using joblib

⚠️ Validation Rules

Study hours: 0–12 hours/day

Sleep hours: 4–12 hours/night

Study + sleep ≤ 18 hours/day

Scores limited to realistic ranges

📦 Important Notes

❌ venv/ is NOT pushed to GitHub

✅ Dependencies managed via requirements.txt

🧪 Model is pre-trained (no retraining on server start)

🎯 Learning Objectives (For Students)

This project demonstrates:

ML model training & serialization

Feature engineering basics

Django REST APIs

Frontend–backend integration

Virtual environment & dependency management

Real-world ML deployment workflow

🧑‍💻 Author

Student ML Project
Built as part of Project-Based Learning (PBL)
Rwanda Coding Academy

📜 License

This project is for educational purposes.
