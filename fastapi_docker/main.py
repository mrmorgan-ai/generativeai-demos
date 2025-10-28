from model_starter import model_pipeline
from fastapi import FastAPI
from typing import Union

app = FastAPI()

@app.get("/")
def read_root():
    return{"hello": "world"}

@app.get("/items/{item_id}") # type: ignore
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
