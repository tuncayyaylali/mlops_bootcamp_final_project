from fastapi import FastAPI, Request
from schemas import Daily_Electricity, Hourly_Electricity
from scipy.stats import ks_2samp
import pandas as pd
import joblib
import helpers
import prediction_preparation

# Importing Model
prediction_model = joblib.load("saved_models/LightGBM_Model.pkl")
prediction_dataset = pd.read_csv("datasets/GercekZamanliTuketim-01012017-31122021.csv", header=0, \
                           quotechar='"', \
                           delimiter=",", \
                           parse_dates=[['Tarih', 'Saat']], \
                           thousands=".", \
                           decimal=",",
                           encoding="windows-1254")


# FastAPI
app = FastAPI()

# Daily Prediction Function
def make_daily_prediction(model, df, request):
    # Prediction Inputs
    Start_Date = request["Start_Date"]
    Hour = request["Hour"]   
    Number_of_Days = request["Number_of_Days"]
    tahmin = prediction_preparation.prediction(model, df, Start_Date, Hour, Number_of_Days) 
    return tahmin


# Hourly Prediction Function
def make_hourly_prediction(model, df, request):
    # Prediction Inputs
    Start_Date = request["Start_Date"]
    Hour = request["Hour"]   
    Number_of_Days = 0
    tahmin = prediction_preparation.prediction(model, df, Start_Date, Hour, Number_of_Days) 
    return tahmin

# Drift Detecton Function
def detect_drift(data1, data2):
    ks_result = ks_2samp(data1, data2)
    if ks_result.pvalue < 0.05:
        return "Drift exits"
    else:
        return "No drift"
    
@app.post("/daily_prediction")
def predict_electricy_consumption(request: Daily_Electricity):
    prediction = make_daily_prediction(prediction_model, prediction_dataset, request.dict())
    return prediction

@app.post("/hourly_prediction")
def predict_electricy_consumption(request: Hourly_Electricity):
    prediction = make_hourly_prediction(prediction_model, prediction_dataset, request.dict())
    return prediction

# Get Client Info
@app.get("/client")
def client_info(request: Request):
    client_host = request.client.host
    client_port = request.client.port
    return {"client_host": client_host,
            "client_port": client_port}

# Detect Drift
@app.post("/drift")
async def detect(request: Daily_Electricity):

    # Prediction Dataset
    prediction_df, _ = make_daily_prediction(prediction_model, prediction_dataset, request.dict())

    # Drift Detection
    drift = detect_drift(prediction_dataset["Tüketim Miktarı (MWh)"], prediction_dataset["Tüketim Miktarı (MWh)"])
    
    return {"drift": drift}