# Importing Libraries
import warnings
import pandas as pd
import os
import numpy as np
import mlflow
import lightgbm as lgb
import io, logging
import datetime
import boto3
from urllib.parse import urlparse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from dotenv import load_dotenv
from datetime import timedelta
from mlflow.tracking import MlflowClient
import helpers
import eda
import feature_importance

# Settings
load_dotenv(dotenv_path=".env")
warnings.filterwarnings('ignore')

# S3 Connection
s3_res = boto3.resource('s3', aws_access_key_id=os.getenv("aws_access_key_id"), aws_secret_access_key=os.getenv("aws_secret_access_key"))
client = boto3.client('s3', aws_access_key_id=os.getenv("aws_access_key_id"), aws_secret_access_key=os.getenv("aws_secret_access_key"))

# Dataset Load Function From S3
def load_df_from_s3(bucket, key, s3_client, index_col=None, usecols=None, sep=","):
    try:
        logging.info(f"Loading {bucket, key}")
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'], header=0, \
                           quotechar='"', \
                           delimiter=",", \
                           parse_dates=[['Tarih', 'Saat']], \
                           thousands=".", \
                           decimal=",")
    except Exception as e:
        raise logging.exception(e)

# Loading Dataset From S3
df = load_df_from_s3(bucket="vbo-mlops-bootcamp-ty", key="datasets/electricity-consumption.csv", s3_client=client)

# EDA
eda.check_df(df)

# Feature Engineering
df = helpers.main(df)

# Hold-Out
train = df.loc[(df["Tarih_Saat"] < "2021-01-01 00:00:00"), :]
test = df.loc[(df["Tarih_Saat"] > "2020-12-31 23:00:00"), :]
cols = [col for col in df.columns if col not in ["Tarih_Saat", "Tüketim Miktarı (MWh)", "year"]]

X_train = train[cols]
y_train = train["Tüketim Miktarı (MWh)"]

X_test = test[cols]
y_test = test["Tüketim Miktarı (MWh)"]

# LightGBM Model
lgbtrain = lgb.Dataset(data=X_train, label=y_train, feature_name=cols)
lgbtest = lgb.Dataset(data=X_test, label=y_test, reference=lgbtrain, feature_name=cols)

# Hyperparamaters for LightGBM Model
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 500,
              'nthread': -1}

# Cost Function for LightGBM Model
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# Fitting and Performance Metrics of LightGBM Model 
model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbtest],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

# Performance Metrics after Hyperparameter Optimization
y_pred_val = model.predict(X_test, num_iteration=model.best_iteration)
smape(np.expm1(y_pred_val), np.expm1(y_test))

# Hyperparameters for Final LightGBM Model
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

# Dataset for Final LightGBM Model
X_all = pd.concat([X_train, X_test], axis=0)
y_all = pd.concat([y_train, y_test], axis=0)
lgbtrain_all = lgb.Dataset(data=X_all, label=y_all, feature_name=cols)

# Fitting and Performance Metrics of Final LightGBM Model
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
final_pred_val = final_model.predict(X_all, num_iteration=model.best_iteration)
smape(np.expm1(final_pred_val), np.expm1(y_all))

# Saving Model Object
import joblib
joblib.dump(final_model, "saved_models/LightGBM_Model.pkl")

# Feature Importance
feature_importance.plot_lgb_importances(model, plot=True)