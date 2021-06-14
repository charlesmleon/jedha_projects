from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check if request has a JSON content
    if request.json:
        # Get the JSON as dictionnary
        req = request.get_json()
        # Check mandatory key
        if "input" in req.keys():
            # Load model
            classifier = joblib.load("models/model.joblib")
            # Predict
            winedata = req["input"] #input should be a list of x-values
            print('winedata: ', winedata)
            prediction = classifier.predict(winedata)
            prediction = prediction.tolist()
            print('prediction: ', prediction, 'type', type(prediction))
            # Return the result as JSON but first we need to transform the
            # result so as to be serializable by jsonify()
            return jsonify({"prediction": prediction}), 200
    return jsonify({"msg": "Error: not a JSON or no input key in your request"})


if __name__ == "__main__":
    app.run(debug=True)