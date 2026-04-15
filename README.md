 Stroke Prediction Using Hybrid Deep Learning (MLP + Autoencoder)
Project Overview

This project focuses on predicting the likelihood of stroke using a hybrid deep learning model that combines Autoencoder and Multi-Layer Perceptron (MLP). The system analyzes patient health data and provides predictions to assist in early diagnosis.

Objective
Build a predictive model for stroke detection
Improve accuracy using hybrid deep learning techniques
Develop a simple web interface for predictions
Dataset

The dataset contains patient health records with the following features:

Age
Gender
Hypertension
Heart Disease
Average Glucose Level
BMI
Smoking Status
Technologies Used
Python
Pandas
NumPy
Scikit-learn
TensorFlow / Keras
Flask
HTML, CSS
Model Architecture

Autoencoder is used for feature extraction and dimensionality reduction.
MLP (Multi-Layer Perceptron) is used for classification of stroke and non-stroke cases.

Workflow
Data preprocessing
Feature encoding and scaling
Autoencoder training
Feature extraction
MLP model training
Prediction
Web Application

The application includes:

User login
Stroke prediction page
Health tips section
How to Run the Project
Clone the Repository
git clone https://github.com/your-username/stroke-prediction.git
cd stroke-prediction
Install Requirements
pip install -r requirements.txt
Run the Application
python app.py
Open in Browser

http://127.0.0.1:5000/

Results

The hybrid model provides better prediction performance compared to basic machine learning models.

Project Structure
stroke-prediction/
│── app.py
│── model/
│   ├── autoencoder.h5
│   ├── mlp_model.h5
│── templates/
│── static/
│── dataset/
│── requirements.txt
│── README.md
Limitations
Depends on dataset quality
Not a replacement for professional medical advice
Future Improvements
Use larger and more diverse datasets
Improve model performance
Deploy the application online
Add real-time prediction features
Author

Your Name

License

This project is for academic and educational purposes only.
