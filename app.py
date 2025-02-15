from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model/house_price_model.h5")

@app.route('/')  # <-- This is missing in your code
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['income'], data['house_age'], data['rooms'], 
                          data['bedrooms'], data['population']]])
    prediction = model.predict(features)
    return jsonify({'predicted_price': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
