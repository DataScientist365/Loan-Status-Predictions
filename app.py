from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("loan_prediction.pkl", "rb")) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST']) 
def predict(): 
       int_features = [int(x) for x in request.form.values()] 
       final_features = [np.array(int_features)] 
       prediction = model.predict(final_features)
       if int(prediction)== 1: 
            prediction ='Loan Status is Default'
       else: 
            prediction ='Loan Status is Non-Default'            
       return render_template("index.html", prediction = prediction) 
     


if __name__ == "__main__":
    app.run(debug=True)
