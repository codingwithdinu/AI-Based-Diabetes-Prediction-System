# 🩺 AI-Based Diabetes Prediction System

## 📌 Project Overview
Diabetes is a chronic disease that can lead to severe health complications if not detected early. This project presents an AI-based machine learning system designed to predict the likelihood of diabetes using key medical and demographic parameters. The system supports early diagnosis and aligns with the United Nations Sustainable Development Goal 3 (Good Health and Well-Being).

---

## 🎯 Objective
- To develop a machine learning model for early diabetes prediction  
- To reduce dependence on invasive diagnostic procedures  
- To provide a fast, cost-effective, and reliable screening tool  
- To promote preventive healthcare using AI  

---

## 🌍 Sustainable Development Goal
**SDG 3: Good Health and Well-Being**  
Ensure healthy lives and promote well-being for all at all ages.

---

## 🧠 Technologies Used
- Python  
- Machine Learning (Support Vector Machine)  
- NumPy, Pandas  
- Scikit-learn  
- Streamlit  
- Jupyter Lab  

---

## 📊 Dataset
- PIMA Indians Diabetes Dataset  
- Includes medical and demographic attributes such as glucose level, BMI, blood pressure, insulin, and age.

---

## ⚙️ Machine Learning Model
- Algorithm: Support Vector Machine (SVM)
- Data Standardization: StandardScaler
- Model Evaluation Metric: Accuracy Score

---

## 📈 Model Performance
- Training Accuracy: ~78%
- Testing Accuracy: ~77%
- The model shows consistent performance with minimal overfitting.

---

## 🌐 Live Application (Deployment)

The diabetes prediction system is deployed using Streamlit Community Cloud.

🔗 Live App Link:
👉 https://ai-based-diabetes-prediction-system.streamlit.app/

### ▶️ How to Use the Live App

Open the deployment link in any web browser.
Enter medical parameters such as:
- Pregnancies
- Glucose level
- Blood Pressure
- BMI
- Age
Click the Predict button.
The system will display whether the person is Diabetic or Non-Diabetic.
✅ No installation required
✅ Accessible on mobile and desktop

### ▶️ Run the Application Locally

To run the application on your local system:
- pip install -r requirements.txt
- streamlit run app.py

### 📁 Project Structure
- ├── app.py
- ├── requirements.txt
- ├── diabetes_model.pkl
- ├── scaler.pkl
- ├── Untitled.ipynb
- ├── diabetes.csv
- └── README.md

### 🧪 Sample Input for Testing
- Pregnancies: 6
- Glucose: 160
- Blood Pressure: 80
- Skin Thickness: 35
- Insulin: 180
- BMI: 35.2
- Diabetes Pedigree Function: 0.65
- Age: 50

## 🔮 Future Enhancements

- Integration with mobile healthcare applications
- Cloud deployment using IBM Cloud
- Real-time monitoring using IoT devices
- Extension to predict other chronic diseases

## 📝 Conclusion

This project demonstrates the effective application of Artificial Intelligence and Machine Learning in healthcare. By enabling early detection of diabetes, the system improves patient outcomes and aligns with SDG 3 – Good Health and Well-Being.

---

