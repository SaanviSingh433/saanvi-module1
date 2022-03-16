from flask import Flask , render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import tree
from sklearn.svm import SVR
import xgboost as xgb
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
app = Flask(__name__,template_folder="Template")
@app.route('/')
def login():
    return render_template("mainfile.html")
@app.route('/navigate_price', methods=['GET'])

def login_price():
    return render_template("price.html")

@app.route('/verify_price', methods=['POST'])
def verfiy_price():
    request_state= request.form["state"]
    request_district= request.form["district"]
    request_market= request.form["market"]
    request_commodity= request.form["commodity"]
    request_price= request.form["min_price"]
    


    data = pd.read_csv('price_prediction_data.csv')
    #print(data)

    # missing data
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        #print(missing_data)

    #request_state="Assam"
    #request_district="cachar"
    #request_market="cachar"
    #request_price="5200"

    encoder= preprocessing.LabelEncoder()
    state=encoder.fit_transform(data['state'])
    request_price_mapping=0
    for i,items in enumerate(encoder.classes_):
        #print(items,"->",i)
        if items==request_state:
            request_state_mapping=i
            break
        
    district=encoder.fit_transform(data['district'])
    request_district_mapping=0
    for i,items in enumerate(encoder.classes_):
        #print(items,"->",i)
        if items==request_district:
            request_district_mapping=i
            break
        
    market=encoder.fit_transform(data['market'])
    request_market_mapping=0
    for i,items in enumerate(encoder.classes_):
        #print(items,"->",i)
        if items==request_market:
            request_market_mapping=i
            break
        
    min_price=encoder.fit_transform(data['min_price'])
    request_price_mapping=0
    for i,items in enumerate(encoder.classes_):
        #print(items,"->",i)
        if items==request_price:
            request_price_mapping=i
            break
    
    commodity=encoder.fit_transform(data['commodity'])
    request_commodity_mapping=0
    for i,items in enumerate(encoder.classes_):
        #print(items,"->",i)
        if items==request_price:
            request_commodity_mapping=i
            break
        
        #modal_price = encoder.fit_transform(data['modal_price'])
    state = np.array(state)
    district = np.array(district)
    market = np.array(market)
    min_price = np.array(min_price)
    commodity = np.array(commodity)
    model = np.array(data['modal_price'])
    feature_for_prediction = list(zip(state,district,market,commodity,min_price))
    labels = data['modal_price']
    target = list(model)


    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(feature_for_prediction,target,test_size = 0.2)

    best = 0
    #for _ in range(100):
    #    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(feature_for_prediction,target,test_size = 0.2)

     #   model = RandomForestRegressor(n_estimators=15)
     #   model.fit(x_train, y_train)
      #  score = model.score(x_test, y_test)
     #   if score > best:
     #       best = score
     #       with open("random_tree.pickle", "wb") as f:
      #          pickle.dump(model, f)

    pickle_load = open("random_tree.pickle", "rb")
    random = pickle.load(pickle_load)
    score = random.score(x_test, y_test)


    print("Random Forest : ", score)
        #predicting values
    #user_input = [request_state_mapping,request_district_mapping,request_market_mapping,request_price_mapping]
    user_input = [request_state_mapping,request_district_mapping,request_market_mapping,request_commodity_mapping,request_price_mapping]
    user_input = np.array(user_input)
    user_input = user_input.reshape(1, -1)
    prediction = random.predict(user_input)

    print(prediction)
    return render_template("test_price.html", prediction=prediction)


if __name__ == '__main__' :
    app.run()
print("richa")
