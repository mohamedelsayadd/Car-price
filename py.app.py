from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
# load the model
model = pickle.load(open('car_price.pkl', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('car.html', **locals())

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                             data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
