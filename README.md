# Credit Card Fraud Detection


## 📌 Project Overview


Credit Card Fraud Detection is an AI/ML project that detects potentially fraudulent credit card transactions using Machine Learning.


The project uses transaction data to train a Random Forest classification model and provides a Flask-based web application where users can enter transaction details and receive a fraud prediction.


## 🎯 Objectives


- Analyze credit card transaction data
- Perform data preprocessing and feature engineering
- Handle categorical and numerical features
- Train a Machine Learning classification model
- Evaluate the fraud detection model
- Build a Flask web application
- Provide real-time fraud prediction through a web interface


## 🛠️ Technologies Used


- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Flask
- HTML
- CSS
- JavaScript
- Joblib
- Git & GitHub


## 🤖 Machine Learning Model


The project uses:


**Random Forest Classifier**


The model is trained to classify transactions into:


- `0` → Legitimate Transaction
- `1` → Fraudulent Transaction


## 🔄 Project Workflow


```text
Dataset
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Train / Validation Split
   ↓
Random Forest Model
   ↓
Model Evaluation
   ↓
Save Model
   ↓
Flask Web Application
   ↓
Fraud Prediction
📂 Project Structure
Credit-Card-Fraud-Detection/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── images/
│
├── models/
│   ├── fraud_model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── reports/
│   └── model_evaluation.txt
│
├── src/
│   ├── preprocess.py
│   └── train_model.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
🌐 Web Application

The project includes a Flask web application.

Users can enter transaction information through the web interface and receive a prediction indicating whether the transaction is potentially fraudulent or legitimate.

Run the Application

Activate the virtual environment:

.venv\Scripts\activate

Install the required packages:

pip install -r requirements.txt

Run the Flask application:

python app.py

Open the application in your browser:

http://127.0.0.1:5000
📊 Dataset

The dataset contains credit card transaction information including transaction amount, category, location-related information, and other transaction features.

The dataset contains a highly imbalanced target variable, with fraudulent transactions representing a small percentage of total transactions.

🔍 Features Used

The model uses features including:

Transaction Amount
Category
Gender
City Population
Latitude
Longitude
Merchant Latitude
Merchant Longitude
Transaction Hour
Transaction Day
Transaction Month
Transaction Day of Week
📈 Model Evaluation

The model is evaluated using classification metrics including:

Precision
Recall
F1-Score
Accuracy
Confusion Matrix

The detailed evaluation results are available in:

reports/model_evaluation.txt
⚠️ Important Note

This project is developed for educational and demonstration purposes. It should not be used as a production financial fraud detection system without additional validation, security, monitoring, and domain-specific testing.

👨‍💻 Author

Sathish R

Computer Science & Engineering Student