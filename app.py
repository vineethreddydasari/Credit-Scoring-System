from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("credit_scoring_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    sex = float(request.form["sex"])
    job = float(request.form["job"])
    housing = float(request.form["housing"])
    saving = float(request.form["saving"])
    checking = float(request.form["checking"])
    credit_amount = float(request.form["credit_amount"])
    duration = float(request.form["duration"])
    purpose = float(request.form["purpose"])

    data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "Job": [job],
        "Housing": [housing],
        "Saving accounts": [saving],
        "Checking account": [checking],
        "Credit amount": [credit_amount],
        "Duration": [duration],
        "Purpose": [purpose]
    })

    prediction = model.predict(data)[0]

    try:
        probability = model.predict_proba(data)

        if prediction == 1:
            approval_percentage = round(probability[0][1] * 100, 2)
        else:
            approval_percentage = round(probability[0][0] * 100, 2)

    except:
        approval_percentage = "Not Available"

    if prediction == 1:
        result = "✅ Good Credit Risk"
    else:
        result = "⚠️ Bad Credit Risk"

    return render_template(
        "index.html",
        prediction_text=result,
        approval_percentage=approval_percentage
    )


if __name__ == "__main__":
    app.run(debug=True)