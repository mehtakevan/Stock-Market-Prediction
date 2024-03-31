from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pydantic import BaseModel

def download_stock_data(symbol, period="6mo"):
    ticker = yf.Ticker(symbol)
    stock_data = ticker.history(period=period)
    return stock_data

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data.to_frame())
    return dataset, scaler

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length, 0])
        y.append(data[i + seq_length, 0])
    return np.array(x), np.array(y)

def build_lstm_model(seq_length):
    model = Sequential()
    model.add(LSTM(60, input_shape=(seq_length, 1), return_sequences=True))
    model.add(LSTM(30, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, batch_size=1, epochs=5):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return model

# Pydantic model for request body
class Stock(BaseModel):
    symbol: str

class Prediction(BaseModel):
    price1:float
    price2:float
    price3:float
    price4:float
    price5:float


def run_lstm_model(symbol, seq_length=60, period="6mo"):
    # Download stock data
    stock_data = download_stock_data(symbol, period)

    # Prepare data
    dataset, scaler = prepare_data(stock_data['Close'])

    # Splitting the data into training and testing sets
    train_data_len = round(len(dataset) * 0.8)
    train_data = dataset[0:train_data_len, :]
    test_data = dataset[train_data_len - seq_length:, :]

    # Creating sequences for LSTM
    x_train, y_train = create_sequences(train_data, seq_length)
    x_test, y_test = create_sequences(test_data, seq_length)

    # Reshaping the data for LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Building and training the LSTM model
    model = build_lstm_model(seq_length)
    model = train_model(model, x_train, y_train)

    pred = model.predict(x_test)
    pred_inverse = scaler.inverse_transform(pred)


    #Predicting Future 5 Days Values

    #Predicting +1 Day from last day of dataset
    x_test_59_1 = x_test[len(x_test)-1][1:60]
    x_test_59_1 = np.append(x_test_59_1, [pred[len(pred)-1]], axis=0)
    x_test_59_1 = x_test_59_1.reshape(1,60,1)
    predp = model.predict(x_test_59_1)

    #Predicting +2 Day from last day of dataset
    x_test_59_2 = x_test_59_1[0][1:60]
    x_test_59_2 = np.append(x_test_59_2, [predp[len(predp)-1]], axis=0)
    x_test_59_2 = x_test_59_2.reshape(1,60,1)
    predpp = model.predict(x_test_59_2)

    #Predicting +3 Day from last day of dataset
    x_test_59_3 = x_test_59_2[0][1:60]
    x_test_59_3 = np.append(x_test_59_3, [predpp[len(predpp)-1]], axis=0)
    x_test_59_3 = x_test_59_3.reshape(1,60,1)
    predppp = model.predict(x_test_59_3)

    #Predicting +4 Day from last day of dataset
    x_test_59_4 = x_test_59_3[0][1:60]
    x_test_59_4 = np.append(x_test_59_4, [predppp[len(predppp)-1]], axis=0)
    x_test_59_4 = x_test_59_4.reshape(1,60,1)
    predpppp = model.predict(x_test_59_4)

    #Predicting +5 Day from last day of dataset
    x_test_59_5 = x_test_59_4[0][1:60]
    x_test_59_5 = np.append(x_test_59_5, [predpppp[len(predpppp)-1]], axis=0)
    x_test_59_5 = x_test_59_5.reshape(1,60,1)
    predppppp = model.predict(x_test_59_5)
    print(type(predp))

    print('+1')
    result = scaler.inverse_transform(predp)
    print('+2')
    result2 = scaler.inverse_transform(predpp)
    print('+3')
    result3 = scaler.inverse_transform(predppp) 
    print('+4')
    result4 = scaler.inverse_transform(predpppp)
    print('+5')
    result5 = scaler.inverse_transform(predppppp)

    preddiction = Prediction(price1=float(result[0]),price2=float(result2[0]),price3=float(result3[0]),price4=float(result4[0]),price5=float(result5[0]))
    return preddiction

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your desired list of origins
    allow_credentials=True,
    allow_methods=["*"],  # Change this to your desired list of HTTP methods
    allow_headers=["*"],  # Change this to your desired list of HTTP headers
)

@app.post("/prediction")
def root(stock:Stock):
    return(run_lstm_model(stock.symbol))

@app.get("/")
def fun():
    return "Hi from stock prediction"
    