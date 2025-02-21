import pandas as pd
import joblib
from flask import Flask, render_template, request
from forms import InputForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"

# Load trained model
model = joblib.load("model.joblib")

# Load dataset to retrieve categorical values
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
X_data = pd.concat([train, val], axis=0).drop(columns="price")

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    form = InputForm()
    message = ""

    if form.validate_on_submit():
        # Extract input values
        user_input = {
            "airline": form.airline.data,
            "date_of_journey": form.date_of_journey.data.strftime("%Y-%m-%d"),
            "source": form.source.data,
            "destination": form.destination.data,
            "dep_time": form.dep_time.data.strftime("%H:%M:%S"),
            "arrival_time": form.arrival_time.data.strftime("%H:%M:%S"),
            "duration": form.duration.data,
            "total_stops": form.total_stops.data,
            "additional_info": form.additional_info.data
        }

        print("Received Data:", user_input)  # Debugging statement

        # Convert to DataFrame
        x_new = pd.DataFrame([user_input])

        # Handle categorical encoding
        categorical_cols = ["airline", "source", "destination", "additional_info"]
        x_new = pd.get_dummies(x_new, columns=categorical_cols)

        # Ensure all required columns exist (matching training data)
        missing_cols = set(X_data.columns) - set(x_new.columns)
        for col in missing_cols:
            x_new[col] = 0  # Fill missing columns with 0

        # Reorder columns to match training data
        x_new = x_new[X_data.columns]

        # Make prediction
        prediction = model.predict(x_new)[0]

        message = f"The predicted price is {prediction:,.0f} INR!"
    else:
        message = "Please provide valid input details!"

    return render_template("predict.html", title="Predict", form=form, output=message)

if __name__ == "__main__":
    app.run(debug=True)
