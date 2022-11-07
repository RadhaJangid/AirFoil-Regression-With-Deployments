
import pickle
from statistics import linear_regression
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
random_model=pickle.load(open('Airfoil_Self_Noise.pkl','rb'))
knn_model=pickle.load(open('Airfoil_Self_Noise1.pkl','rb'))
linear_regression_model=pickle.load(open('Airfoil_Self_Noise2.pkl','rb'))
ridge_regression_model=pickle.load(open('Airfoil_Self_Noise3.pkl','rb'))
lasso_regression_model=pickle.load(open('Airfoil_Self_Noise4.pkl','rb'))
SVR_model=pickle.load(open('Airfoil_Self_Noise5.pkl','rb'))
decision_tree_regression_model=pickle.load(open('Airfoil_Self_Noise6.pkl','rb'))
@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

#@app.route('/predict_api',methods=['POST'])
#def predict_api():

    #data=request.json['data']
    #print(data)
    #new_data=[list(data.values())]
    #output=model.predict(new_data)[0]
    #return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():

    data=[float(x) for x in request.form.values()]

    final_features = [np.array(data)]
    
    
   
    
    linear_output=linear_regression_model.predict(final_features)[0]
    ridge_output=ridge_regression_model.predict(final_features)[0]
    lasso_output=lasso_regression_model.predict(final_features)[0]
    SVR_output=SVR_model.predict(final_features)[0]
    knn_output=knn_model.predict(final_features)[0]
    decision_output=decision_tree_regression_model.predict(final_features)[0]
    output=random_model.predict(final_features)[0]
    
    best_model=max([linear_output,ridge_output,lasso_output,SVR_output,knn_output
    ,decision_output,output])

    
  
    if (knn_output >= decision_output) and (knn_output >= output):
        largest = "KNN Model"
  
    elif (decision_output >= knn_output) and (decision_output >= output):
        largest = "Decision Model"
    else:
        largest = "Random Forest Model"
    
   


    return render_template('home.html', linear_prediction="{}".format(linear_output),
    ridge_prediction="{}".format(ridge_output),
    lasso_prediction="{}".format(lasso_output),
    svr_prediction="{}".format(SVR_output),
    knn_prediction="{}".format(knn_output),
    dt_prediction="{}".format(decision_output),
    random_prediction="{}".format(output),
    frequency="{}".format(data[0]),
    angle_of_attack="{}".format(data[1]),
    chord_length="{}".format(data[2]),
    velocity="{}".format(data[3]),
    side="{}".format(data[4]),
    best_model_result="{}".format(best_model),
    best_model_name="{}".format(largest))
    
    
    
    




if __name__=="__main__":
    app.run(debug=True)
