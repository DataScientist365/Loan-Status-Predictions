from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# prediction function 
def ValuePredictor(to_predict_list): 
    print(np.array(to_predict_list))
    to_predict = np.array(to_predict_list).reshape(1, 33)
    loaded_model = pickle.load(open(r"C:\Users\Mihir\Desktop\Projects\Python Project - Bank Lending\Code File\loan_prediction.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0] 
  
@app.route('/predict', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        predict = ValuePredictor(to_predict_list)         
        if int(predict)== 1: 
            prediction ='Loan Status is Default'
         
        else: 
            prediction ='Loan Status is Non-Default'            
        return render_template("index.html", prediction = prediction) 
     


if __name__ == "__main__":
    app.run(debug=True)
