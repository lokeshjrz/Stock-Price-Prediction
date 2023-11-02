import numpy as np 
import requests
import pandas
from keras.models import load_model
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras import Sequential
from sklearn.model_selection import train_test_split
from flask import render_template


# model = load_model('humanAI.h5') 
def preprocessdata(stock):
    stock = stock.upper()
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-historical-data"

    querystring = {"period1":"1546448400","period2":"1562086800","symbol":stock,"frequency":"1d","filter":"history"}

    headers = {
        "X-RapidAPI-Key": "b5e45be06fmsh8475e309170b38cp137f68jsn080566a5ad58",
        "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring)

        print(response.json())
        a = response.json()
        import pandas as pd
        data = pd.DataFrame(a["prices"])
        data.head()
        data = data.drop(columns=["date","volume","adjclose","amount","type","data"],axis=1)
        data = data.dropna()
        x = data.iloc[:,:-1]
        y = data.iloc[:,-1:]
        X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.5)
        model = Sequential()
        model.add(Dense(10,activation="relu"))
        model.add(Dense(100,activation="relu"))
        model.add(Dense(300,activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
        model.fit(X_train,y_train,epochs = 100)
        y_pred = model.predict(X_test)
        prediction = y_pred[0][0]
        print(prediction)
        return prediction
    except requests.exceptions.RequestException as e:
        print(e)
        return f'An error occurred: {str(e)}'


