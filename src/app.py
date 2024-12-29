'''from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.post("/predict")
def predict_sentiment(text: str):
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return {"sentiment": sentiment}'''
from flask import Flask, request, render_template, jsonify
import joblib
import os

# Load pre-trained model and vectorizer
model_path = "E:/sentiment_ML_PROJECT/models/sentiment_model.pkl"
vectorizer_path = "E:/sentiment_ML_PROJECT/models/vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

#app = Flask(__name__)
app = Flask(__name__, template_folder='E:/sentiment_ML_PROJECT/templates')
# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Get the text input from the user
    input_text = request.form.get("text")

    if not input_text:
        return jsonify({"error": "Please provide text input for sentiment analysis."})

    # Transform the input text using the loaded vectorizer
    transformed_text = vectorizer.transform([input_text])

    # Predict sentiment using the loaded model
    prediction = model.predict(transformed_text)

    # Map prediction to sentiment label
    sentiment = "Positive" if prediction[0] == 1 else "Negative"

    return jsonify({"input_text": input_text, "sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)

