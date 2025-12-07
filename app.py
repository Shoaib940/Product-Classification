from flask import Flask, render_template, request
import pickle
import re
import string
import os

# Initialize Flask app
app = Flask(_name_)

# -----------------------
# Load trained objects
# -----------------------
with open("ridge_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -----------------------
# Preprocessing function (same as training)
# -----------------------
def preprocess_text(text):
    if not isinstance(text, str):
        text = ""
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# -----------------------
# Home route
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        product_text = request.form.get("product_text", "")

        # Clean the text
        cleaned_text = preprocess_text(product_text)

        # Vectorize
        vect_text = vectorizer.transform([cleaned_text])

        # Predict
        pred_encoded = model.predict(vect_text)
        prediction = le.inverse_transform(pred_encoded)[0]

    return render_template("index.html", prediction=prediction)

# -----------------------
# Run app (Railway compatible)
# -----------------------
if _name_ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
