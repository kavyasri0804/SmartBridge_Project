from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
model = pickle.load(open(r"C:\Users\nambu\OneDrive\Desktop\Project\model.pkl", 'rb'))
scaler = pickle.load(open(r"C:\Users\nambu\OneDrive\Desktop\Project\scale.pkl", 'rb'))
FEATURES = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day']
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read and clean input
        raw_values = [request.form.get(f) for f in FEATURES]
        print("🟡 Raw form values:", raw_values)
        input_values = []
        for val in raw_values:
            if val is None or val == '':
                return render_template('index.html', prediction_text="❌ Missing input value.")
            try:
                input_values.append(float(val))
            except ValueError:
                return render_template('index.html', prediction_text="❌ Invalid input: please enter numbers only.")

        print("✅ Cleaned input values:", input_values)
        # Build DataFrame
        df = pd.DataFrame([input_values], columns=FEATURES)
        print("📊 DataFrame before scaling:\n", df)
        # Apply scaler
        df_scaled = scaler.transform(df)
        print("📏 Scaled input:", df_scaled)
        # Make prediction
        prediction = model.predict(df_scaled)
        return render_template('index.html', prediction_text=f"✅ Estimated Traffic Volume: {prediction[0]:,.0f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Unexpected error occurred: {e}")
if __name__ == '__main__':
    app.run(debug=True)