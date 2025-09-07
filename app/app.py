from flask import Flask, render_template, request
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model & tokenizer
MODEL_PATH = "../saved_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH,from_pt=True)

def predict_fake_news(text: str):
    inputs = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128
    )
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    label_id = int(np.argmax(probs))
    return {
        "fake_prob": float(probs[0]),
        "real_prob": float(probs[1]),
        "prediction": "Real News ✅" if label_id == 1 else "Fake News ❌"
    }

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_text = request.form["news_text"]
        result = predict_fake_news(user_text)
        return render_template("result.html", text=user_text, result=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
