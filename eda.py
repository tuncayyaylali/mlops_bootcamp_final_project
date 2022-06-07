import pandas as pd

# EDA Function
def check_df(dataframe, column, target, head=5):
    print("--------------------------------------------------")
    print("EXPLANATORY DATA ANALYSIS")
    print("--------------------------------------------------")
    print("Descriptive Analysis")
    print("--------------------------------------------------")
    print(dataframe[target].describe().T)
    print("--------------------------------------------------")
    print("Head Observations of Dataframe")
    print("--------------------------------------------------")
    print(f"{dataframe.head()}")
    print("--------------------------------------------------")
    print("Tail Observations of Dataframe")
    print("--------------------------------------------------")
    print(f"{dataframe.tail()}")
    print("--------------------------------------------------")
    print(f"Information of Data frame: {dataframe.info()}")
    print("--------------------------------------------------")
    print("MISSING VALUES")
    print("--------------------------------------------------")
    print(f"{dataframe.isnull().sum()}")
    print("--------------------------------------------------")
    print("First Observation Date")
    print("--------------------------------------------------")
    print(dataframe[column].min())
    print("--------------------------------------------------")
    print("Last Observation Date")
    print("--------------------------------------------------")
    print(dataframe[column].max())