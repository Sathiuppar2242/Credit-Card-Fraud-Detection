from flask import Flask, render_template, request
import pandas as pd
import joblib


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

model = joblib.load("models/fraud_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        # ----------------------------------------------------
        # Get values from HTML form
        # ----------------------------------------------------

        amount = float(request.form["amount"])

        category = request.form["category"]

        gender = request.form["gender"]

        city_population = int(
            request.form["city_population"]
        )

        customer_latitude = float(
            request.form["customer_latitude"]
        )

        customer_longitude = float(
            request.form["customer_longitude"]
        )

        merchant_latitude = float(
            request.form["merchant_latitude"]
        )

        merchant_longitude = float(
            request.form["merchant_longitude"]
        )

        transaction_hour = int(
            request.form["transaction_hour"]
        )

        transaction_day = int(
            request.form["transaction_day"]
        )

        transaction_month = int(
            request.form["transaction_month"]
        )

        transaction_day_of_week = int(
            request.form["transaction_day_of_week"]
        )


        # ----------------------------------------------------
        # Create input DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            {
                "amt": [amount],
                "category": [category],
                "gender": [gender],
                "city_pop": [city_population],
                "lat": [customer_latitude],
                "long": [customer_longitude],
                "merch_lat": [merchant_latitude],
                "merch_long": [merchant_longitude],
                "transaction_hour": [transaction_hour],
                "transaction_day": [transaction_day],
                "transaction_month": [transaction_month],
                "transaction_day_of_week": [
                    transaction_day_of_week
                ]
            }
        )


        # ----------------------------------------------------
        # Apply preprocessing
        # ----------------------------------------------------

        processed_data = preprocessor.transform(
            input_data
        )


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        result = model.predict(
            processed_data
        )[0]

        probability = model.predict_proba(
            processed_data
        )[0][1]


        # ----------------------------------------------------
        # Convert prediction to readable text
        # ----------------------------------------------------

        if result == 1:

            prediction = "Potential Fraudulent Transaction"

        else:

            prediction = "Transaction Appears Legitimate"


    # --------------------------------------------------------
    # Render HTML page
    # --------------------------------------------------------

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )