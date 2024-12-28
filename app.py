from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.post("/predict")
def predict_sentiment(text: str):
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return {"sentiment": sentiment}
