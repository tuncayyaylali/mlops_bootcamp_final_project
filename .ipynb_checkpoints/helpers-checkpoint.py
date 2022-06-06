# Importing Libraries
import pandas as pd
import numpy as np
from datetime import timedelta
import datetime as dt

# Feature Extracting Function
def create_date_features(dataframe):
    dataframe['month'] = dataframe["Tarih_Saat"].dt.month
    dataframe['day_of_month'] = dataframe["Tarih_Saat"].dt.day
    dataframe['day_of_year'] = dataframe["Tarih_Saat"].dt.dayofyear
    dataframe['week_of_year'] = dataframe["Tarih_Saat"].dt.isocalendar().week.astype("int32")
    dataframe['day_of_week'] = dataframe["Tarih_Saat"].dt.dayofweek
    dataframe['year'] = dataframe["Tarih_Saat"].dt.year
    dataframe["is_wknd"] = dataframe["Tarih_Saat"].dt.weekday // 4
    dataframe['is_month_start'] = dataframe["Tarih_Saat"].dt.is_month_start.astype(int)
    dataframe['is_month_end'] = dataframe["Tarih_Saat"].dt.is_month_end.astype(int)
    return dataframe


# Random Noise Function
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


# Lag/Shifted Features Function
def lag_features(dataframe):
    for lag in [1, 720, 2160, 8760]:
        dataframe['Tüketim Miktarı (MWh)_lag_' + str(lag)] = dataframe["Tüketim Miktarı (MWh)"].transform(lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


# Rolling Function
def roll_mean_features(dataframe):
    for window in [1,720, 2160, 8760]:
        dataframe['Tüketim Miktarı (MWh)_roll_mean_' + str(window)] = dataframe["Tüketim Miktarı (MWh)"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


# Exponentially Weighted Mean Features Function
def ewm_features(dataframe):
    for alpha in [0.95, 0.9, 0.8, 0.7, 0.5]:
        for lag in [1,120, 2160, 8760]:
            dataframe["Tüketim Miktarı (MWh)_ewm_alpha_" + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe["Tüketim Miktarı (MWh)"].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


# One Hot Encoding
def get_dummies(dataframe):
    df = pd.get_dummies(dataframe, columns=['day_of_week', 'month'])
    return df


# Converting Target Variable to log(1+Target Variable)
def converting_target(dataframe):
    dataframe["Tüketim Miktarı (MWh)"]=np.log1p(dataframe["Tüketim Miktarı (MWh)"].values)
    return dataframe


# Pipeline
def main(dataframe):
    df_create_date_features = create_date_features(dataframe)
    df_lag_features = lag_features(df_create_date_features)
    df_roll_mean_features = roll_mean_features(df_lag_features)
    df_ewm_features = ewm_features(df_roll_mean_features)
    df_get_dummies = get_dummies(df_ewm_features)
    df_converting_target = converting_target(df_get_dummies)
    return df_converting_target


# Main Function
if __name__ == "__main__":
    main()