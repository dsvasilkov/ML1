from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
from starlette.responses import StreamingResponse

app = FastAPI()

model = joblib.load("data/model.pkl")
scaler = joblib.load("data/scaler.pkl")
encoder = joblib.load("data/encoder.pkl")


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    seats: int


class Items(BaseModel):
    objects: List[Item]


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    categorical_features = [*df.select_dtypes(include=['object']).columns, "seats"]
    data_encoded = encoder.transform(df[categorical_features])
    df_encoded = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out([*categorical_features]))
    df_cat = pd.concat([df.drop(columns=categorical_features), df_encoded], axis=1)
    return scaler.transform(df_cat)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame([item.dict()])
    processed_data = preprocess_data(data)
    prediction = model.predict(processed_data)[0]
    return prediction


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, index_col=0)
    processed_data = preprocess_data(df)
    predictions = model.predict(processed_data)

    df["predicted_price"] = predictions
    output = df.to_csv(index=True)

    return StreamingResponse(iter([output]))