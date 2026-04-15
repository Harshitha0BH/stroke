from flask import Flask, render_template, request, redirect, session, url_for
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

app = Flask(__name__)
app.secret_key = "secret"

# Load models
scaler = pickle.load(open('scaler.pkl', 'rb'))
rf = pickle.load(open('mlp.pkl', 'rb'))  # now RandomForest
pca = pickle.load(open('pca.pkl', 'rb'))   # replaces encoder

# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- USER LOGIN ----------------
@app.route('/user_login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        session['user'] = request.form['username']
        return redirect('/predict')
    return render_template('user_login.html')

# ---------------- DOCTOR LOGIN ----------------
@app.route('/doctor_login', methods=['GET', 'POST'])
def doctor_login():
    error_message = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'doctor' and password == 'doctor123':
            session['doctor'] = username
            return redirect('/doctor_dashboard')
        error_message = "Invalid credentials. Use username 'doctor' and password 'doctor123'."
    return render_template('doctor_login.html', error=error_message)

# ---------------- PREDICTION ----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = [
                int(request.form['gender']),
                float(request.form['age']),
                int(request.form['hypertension']),
                int(request.form['heart_disease']),
                int(request.form['ever_married']),
                int(request.form['work_type']),
                int(request.form['Residence_type']),
                float(request.form['avg_glucose_level']),
                float(request.form['bmi']),
                int(request.form['smoking_status'])
            ]
            data = np.array(data).reshape(1, -1)

            # Scale
            data_scaled = scaler.transform(data)

            # Feature reduction (instead of autoencoder)
            data_reduced = pca.transform(data_scaled)

            # Prediction
            prob = rf.predict_proba(data_reduced)[0][1]

            # Graph
            plt.figure()
            plt.bar(['Stroke Risk'], [prob], color=['#B19CD9'])
            plt.ylim(0, 1)
            plt.savefig('static/result.png')
            plt.close()

            # Risk stage
            if prob < 0.3:
                stage = "Low Risk"
                advice = "Maintain healthy lifestyle"
            elif prob < 0.7:
                stage = "Moderate Risk"
                advice = "Consult doctor if symptoms appear"
            else:
                stage = "High Risk"
                advice = "Consult doctor immediately"

            result_url = url_for('static', filename='result.png') + '?v=' + str(int(time.time()))
            return render_template(
                'result.html',
                prob=round(prob, 4),
                stage=stage,
                advice=advice,
                result_url=result_url
            )

        except Exception as e:
            return render_template('predict.html', error=f"Error in input: {str(e)}. Please enter valid values.")

    return render_template('predict.html')

# ---------------- DOCTOR DASHBOARD ----------------
@app.route('/doctor_dashboard')
def doctor_dashboard():
    return render_template('doctor_dashboard.html')

# ---------------- HEALTH TIPS ----------------
@app.route('/health_tips')
def health_tips():
    return render_template('health_tips.html')

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)