#import asyncio
from fastapi import FastAPI
import time

app = FastAPI()

@app.get("/1")
async def awesome_endpoint():
    print("Hello AI PlayGrounds!")
    time.sleep(5)
    print("Bye AI PlayGrounds!")
