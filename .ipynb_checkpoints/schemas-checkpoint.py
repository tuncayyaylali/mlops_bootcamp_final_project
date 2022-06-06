from pydantic import BaseModel


class Daily_Electricity(BaseModel):
    Start_Date: str
    Hour: str
    Number_of_Days: int

    class Config:
        schema_extra = {
            "example": {
                "Start_Date": "2022-01-01",
                "Hour": "00:00:00",
                "Number_of_Days": 5,
            }
        }

class Hourly_Electricity(BaseModel):
    Start_Date: str
    Hour: str

    class Config:
        schema_extra = {
            "example": {
                "Start_Date": "2022-01-01",
                "Hour": "00:00:00",
            }
        }

class Hourly_Electricity_Consumption(BaseModel):
    Start_Date: str
    Hour: str
    Consumption: float

    class Config:
        schema_extra = {
            "example": {
                "Start_Date": "2022-01-01",
                "Hour": "00:00:00",
                "Comsumption": 30000
            }
        }