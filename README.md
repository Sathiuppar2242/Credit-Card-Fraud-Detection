#  Credit Card Fraud Detection

A Machine Learning-based Credit Card Fraud Detection system that analyzes transaction details and predicts whether a transaction is **Fraudulent** or **Legitimate**.

The project includes a complete Machine Learning pipeline, exploratory data analysis, Random Forest classification model, and a Flask-based web application for real-time fraud prediction.

---

## 🎯 Project Objective

The objective of this project is to build an intelligent fraud detection system capable of identifying potentially fraudulent credit card transactions using Machine Learning.

The system processes transaction information, applies the trained Machine Learning model, and provides a prediction through a user-friendly web interface.

---

## ✨ Key Features

- 🔍 Credit card transaction fraud detection
- 🤖 Random Forest Machine Learning model
- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data preprocessing and feature engineering
- ⏰ Time-based transaction features
- 🌐 Flask web application
- ⚡ Real-time transaction prediction
- 📈 Fraud and legitimate transaction classification
- 💾 Saved trained model and preprocessing pipeline
- 📱 User-friendly web interface

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Random Forest Classifier
- Pandas
- NumPy

### Data Analysis & Visualization
- Matplotlib
- Seaborn
- Jupyter Notebook

### Web Development
- Flask
- HTML
- CSS
- JavaScript

### Development Tools
- VS Code
- Git
- GitHub

---

## 📊 Dataset

The project uses a credit card transaction dataset containing transaction, customer, merchant, location, and fraud-related information.

Important columns include:

- Transaction date and time
- Transaction amount
- Merchant
- Transaction category
- Gender
- City population
- Latitude and longitude
- Merchant latitude and longitude
- Fraud label

### Fraud Distribution

The dataset is highly imbalanced, with legitimate transactions significantly outnumbering fraudulent transactions.

This reflects a realistic fraud detection scenario where fraudulent transactions are rare compared to normal transactions.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis
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

🧹 Data Preprocessing

The preprocessing pipeline performs the following operations:

Handles transaction data
Creates time-based features
Converts categorical variables into numerical representations
Selects relevant features
Prepares the data for Machine Learning
Saves the preprocessing pipeline
Time-Based Features

The project creates:

Transaction Hour
Transaction Day
Transaction Month
Transaction Day of Week
🤖 Machine Learning Model

The project uses a Random Forest Classifier for fraud detection.

Selected features include:

Transaction amount
Category
Gender
City population
Customer latitude
Customer longitude
Merchant latitude
Merchant longitude
Transaction hour
Transaction day
Transaction month
Transaction day of week

The trained model is stored in the models directory.

📈 Model Training

The dataset is divided into training and validation datasets.

Training Data
Training samples: 1,037,340
Fraudulent transactions: 6,005
Legitimate transactions: 1,031,335
Validation Data
Validation samples: 259,335
Fraudulent transactions: 1,501
Legitimate transactions: 257,834

The Random Forest model is trained using the training dataset and evaluated using the validation dataset.

🌐 Web Application

A Flask-based web application is included in this project.

Users can enter transaction information through the web interface, and the trained Machine Learning model predicts whether the transaction is:

🟢 Legitimate

or

🔴 Fraudulent

The application loads the trained model and preprocessing pipeline to perform predictions.

📸 Application Screenshots
Home Page

Fraud Detection Result

📁 Project Structure
Credit-Card-Fraud-Detection/
│
├── data/
│   └── Dataset files
│
├── images/
│   ├── home.png
│   └── fraud-result.png
│
├── models/
│   ├── fraud_model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── Exploratory Data Analysis
│
├── reports/
│   └── Model evaluation reports
│
├── src/
│   ├── preprocess.py
│   └── train_model.py
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   └── index.html
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
⚙️ Installation

Clone the repository:

git clone https://github.com/Sathiuppar2242/Credit-Card-Fraud-Detection.git

Navigate to the project directory:

cd Credit-Card-Fraud-Detection

Create a virtual environment:

python -m venv .venv

Activate the virtual environment on Windows:

.venv\Scripts\Activate.ps1

Install the required dependencies:

pip install -r requirements.txt
▶️ Run the Web Application

Start the Flask application:

python app.py

Then open the application in your browser:

http://127.0.0.1:5000

Enter the transaction details and submit the form to receive a fraud prediction.

🧪 Project Components
1. Data Preprocessing

src/preprocess.py

Handles data preprocessing and feature engineering.

2. Exploratory Data Analysis

notebooks/

Contains analysis and visualization of transaction data and fraud patterns.

3. Model Training

src/train_model.py

Trains and evaluates the Random Forest fraud detection model.

4. Web Application

app.py

Provides the Flask web interface and performs fraud predictions using the trained model.

🔮 Future Enhancements
Improve fraud detection performance using advanced Machine Learning algorithms
Handle class imbalance using advanced techniques
Add model comparison
Add authentication
Deploy the application to a cloud platform
Add transaction history and analytics
Add interactive fraud monitoring dashboard
👨‍💻 Author

Sathish R

Computer Science & Engineering Student

GitHub:
https://github.com/Sathiuppar2242
## Project Features

- Credit card transaction fraud detection
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Fraud and non-fraud distribution analysis
- Time-based transaction features
- Random Forest classification model
- Train-validation data splitting
- Model evaluation using classification metrics
- Saved trained model and preprocessor


## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook
- Git and GitHub


## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook
- Git and GitHub


## Dataset Information

The project uses credit card transaction data for fraud detection.

- Transaction records include transaction amount, category, location, customer information, and time-based features.
- The dataset contains both fraudulent and legitimate transactions.
- Fraud detection is treated as a binary classification problem.
- The dataset is highly imbalanced, with fraudulent transactions representing a small portion of the total transactions.


## Machine Learning Model

The project uses a Random Forest Classifier to identify potentially fraudulent credit card transactions.

### Model Workflow

1. Load and preprocess transaction data
2. Perform feature engineering
3. Split data into training and validation sets
4. Train the Random Forest model
5. Evaluate model performance using classification metrics
6. Save the trained model using Joblib


## Project Results

- The Random Forest classifier is trained to identify fraudulent transactions.
- The project handles highly imbalanced transaction data.
- Time-based transaction features are included to improve fraud detection.
- Model performance is evaluated using classification metrics.
- The trained model and preprocessing pipeline are saved for future predictions.


## Fraud Detection Statistics

The dataset contains a highly imbalanced distribution between legitimate and fraudulent transactions.

- Legitimate transactions: 1,289,169
- Fraudulent transactions: 7,506
- Total training transactions: 1,296,675
- Fraudulent transactions represent approximately 0.58% of the training data.


## Fraud Detection Statistics

The dataset contains a highly imbalanced distribution between legitimate and fraudulent transactions.

- Legitimate transactions: 1,289,169
- Fraudulent transactions: 7,506
- Total training transactions: 1,296,675
- Fraudulent transactions represent approximately 0.58% of the training data.


## Model Training Details

The Random Forest model is trained using selected transaction and time-based features.

### Selected Features

- Transaction amount
- Transaction category
- Gender
- City population
- Latitude and longitude
- Merchant latitude and longitude
- Transaction hour
- Transaction day
- Transaction month
- Transaction day of week

The data is divided into training and validation sets before model training and evaluation.


## Model Training Details

The Random Forest model is trained using selected transaction and time-based features.

### Selected Features

- Transaction amount
- Transaction category
- Gender
- City population
- Latitude and longitude
- Merchant latitude and longitude
- Transaction hour
- Transaction day
- Transaction month
- Transaction day of week

The data is divided into training and validation sets before model training and evaluation.


## Project Workflow

1. Collect and load transaction data
2. Explore transaction and fraud patterns
3. Clean and preprocess the data
4. Create transaction time-based features
5. Select relevant features for model training
6. Split the data into training and validation sets
7. Train the Random Forest classifier
8. Evaluate the model using classification metrics
9. Save the trained model and preprocessor
10. Use the saved model for future fraud predictions

