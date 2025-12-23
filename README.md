# 🛡️ Credit Card Fraud Detection Using Machine Learning

This project implements a **Credit Card Fraud Detection System** using Machine Learning techniques to identify fraudulent transactions from real-world financial data. The system handles highly imbalanced data and provides real-time predictions through a responsive web application.

---

## 📌 Project Overview

- **Problem Type:** Binary Classification (Fraud / Normal)
- **Domain:** Finance, Security, Machine Learning
- **Dataset:** Credit Card Transactions (Anonymized)
- **Deployment:** Streamlit Web Application

---

## 🎯 Objectives

- Detect fraudulent credit card transactions accurately
- Handle class imbalance effectively
- Evaluate models using appropriate performance metrics
- Deploy a user-friendly and responsive prediction system

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  
- **Imbalanced Data Handling:** SMOTE (imbalanced-learn)  
- **Models:** Isolation Forest, Logistic Regression  
- **Model Saving:** Joblib  
- **Web Framework:** Streamlit  
- **IDE:** Visual Studio Code  

---

## 📂 Project Structure

fraud-detection-ml/
│
├── data/
│ └── creditcard.csv
│
├── src/
│ └── train_model.py
│
├── models/
│ ├── fraud_model.pkl
│ └── scaler.pkl
│
├── app/
│ └── app.py
│
├── requirements.txt
└── README.md

--- 

## ⚙️ Steps to Run the Project

### 1️⃣ Install Dependencies

pip install -r requirements.txt
### 2️⃣ Train the Model

python src/train_model.py
### 3️⃣ Run the Web Application

streamlit run app/app.py

### 📊 Model Evaluation Metrics
- Precision
- Recall
- F1-score
- ROC-AUC Score
- Confusion Matrix

⚠️ Accuracy is not used as the primary metric due to severe class imbalance.

### 🌐 Web Application Features
- Real-time fraud prediction
- Fraud probability score
- Interactive risk visualization
- Responsive UI with smooth animations

## ✅ Results
The model successfully identifies fraudulent transactions with improved recall and precision after handling data imbalance using SMOTE.

### 🚀 Future Enhancements
- Integration with real-time transaction APIs

- Use of advanced models like XGBoost

- Feature importance visualization

- Cloud deployment (AWS / Azure / GCP)

📄 License
- This project is developed for educational and internship purposes only

