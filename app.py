import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

# Home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve the feature values from the form
        feature1 = float(request.form['wc'])
        feature2 = float(request.form['bu'])
        feature3 = float(request.form['bgr'])
        feature4 = float(request.form['sc'])
        feature5 = float(request.form['pcv'])
        feature6 = float(request.form['al'])
        feature7 = float(request.form['hemo'])
        feature8 = float(request.form['age'])
        feature9 = float(request.form['su'])
        feature10 = float(request.form['htn'])

        
        
        xgb=joblib.load('xgb_model.pkl') 
        logreg_stage=joblib.load('logistic_regression_model.pkl')  
        
        # Prepare the feature values as input for prediction
        new_patient_features = [[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10]]  # Add more feature values as per your dataset
        
        # Predict CKD presence using random forest
        ckd_prediction = xgb.predict(new_patient_features)
        
        if ckd_prediction == 0:
            # Predict CKD stage using logistic regression
            ckd_stage_prediction = logreg_stage.predict(new_patient_features)
            ckd_stage = ckd_stage_prediction[0]
            return render_template('index.html', prediction=True, stage=ckd_stage)
        else:
            return render_template('index.html', prediction=False)
    else:
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

