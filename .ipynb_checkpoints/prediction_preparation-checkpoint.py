# Importing Libraries
import pandas as pd
import helpers
import joblib
from datetime import timedelta
import numpy as np


# Prediction Preparation Function
def check(start,end):
    Result = []
    Result.append(start)
    start = start + timedelta(hours=1)        
    while start <= end:
        Result.append(start)
        start = start + timedelta(hours=1)
    return Result


# Prediction
def prediction(model, df, Start_Date, Hour, Number_of_Days=0):
    Start_Date = Start_Date+" "+Hour
    Start_Date = pd.to_datetime(Start_Date)
    Main_Date = pd.to_datetime("2022-01-01 00:00:00")
    Finish_Date = Start_Date + timedelta(days=Number_of_Days)
    Finish_Date = pd.to_datetime(Finish_Date)
    prediction_interval = pd.DataFrame(check(Main_Date, Finish_Date), columns=["Tarih_Saat"])
    prediction_interval["Tüketim Miktarı (MWh)"]=np.nan
    prediction_total = pd.concat([df, prediction_interval], axis=0, ignore_index=True)
    prediction_dataset = helpers.main(prediction_total)
    cols = [col for col in prediction_dataset.columns if col not in ["Tarih_Saat", "Tüketim Miktarı (MWh)", "year"]]
    X_predict = prediction_dataset[cols]
    predict = pd.DataFrame(np.expm1(model.predict(X_predict)), columns=["Tüketim Miktarı (MWh)"])
    prediction_interval = pd.concat([prediction_total["Tarih_Saat"], predict], axis=1, ignore_index=True)    
    prediction_interval = prediction_interval.loc[prediction_interval[0] >= Start_Date, :]
    return prediction_interval, prediction_interval[1].sum()