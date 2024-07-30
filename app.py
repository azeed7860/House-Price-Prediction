from flask import Flask,jsonify, request, render_template
import pickle
import numpy as np


flask_app = Flask(__name__)
model = pickle.load(open("reg_model (1).pkl", "rb"))


# Load the trained model
#with open('reg_model.pkl', 'rb') as model_file:
    #model = pickle.load(model_file)

@flask_app.route("/")
def Home():
    return render_template('index.html')

@flask_app.route("/predict", methods=['POST'])

def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])

        return render_template('index.html', prediction_text='The predicted house price is:{}'.format(prediction))

if __name__ == '__main__':
    flask_app.run(debug=True)
