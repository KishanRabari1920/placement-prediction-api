<<<<<<< HEAD
# Student Placement Prediction API

A Machine Learning + FastAPI project that predicts whether a student will be placed based on academic and skill-related features.

---

# Features

* Machine Learning based placement prediction
* FastAPI REST API integration
* Multiple ML model comparison
* Data preprocessing and scaling
* Real-time prediction using API
* Swagger UI testing support
* Beginner-friendly project structure

---

# Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn

---

# Machine Learning Models Used

* Logistic Regression
* Random Forest Classifier
* Support Vector Machine (SVM)

---

# Project Structure

```bash
placement_prediction_project/
│
├── app/
│   └── main.py
│
├── model/
│   ├── train_model.py
│   └── placement_model.pkl
│
├── data/
│   └── students.csv
│
├── requirements.txt
└── README.md
```

---

# Dataset Features

The model uses the following features:

* CGPA
* IQ
* Communication Skills
* Internship Experience

Target Column:

* placed

---

# Installation

## Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

## Move into Project Folder

```bash
cd placement_prediction_project
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train Machine Learning Model

Move to model folder:

```bash
cd model
```

Run:

```bash
python train_model.py
```

This will:

* Train ML models
* Compare accuracies
* Save the best model automatically

---

# Run FastAPI Server

Move to app folder:

```bash
cd app
```

Run server:

```bash
uvicorn main:app --reload
```

---

# API URLs

## Home Route

```bash
http://127.0.0.1:8000/
```

## Swagger Documentation

```bash
http://127.0.0.1:8000/docs
```

---

# Sample API Request

```json
{
  "cgpa": 8.5,
  "iq": 120,
  "communication": 9,
  "internship": 1
}
```

---

# Sample API Response

```json
{
  "prediction": "Placed",
  "confidence_percentage": 92.45
}
```

---

# Concepts Used

* Data Preprocessing
* Feature Scaling
* Classification Models
* Model Evaluation
* REST API Development
* Model Serialization using Pickle
* FastAPI Backend Integration

---

# Future Improvements

* Add Frontend UI
* Database Integration
* User Authentication
* Docker Support
* Cloud Deployment
* Model Monitoring
* CI/CD Pipeline

---

# Author

Kishan Rabari

---
=======
# placement-prediction-api
>>>>>>> a0fd420d40bae765ef1602368785862296fd82ce
