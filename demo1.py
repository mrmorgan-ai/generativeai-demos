import asyncio
#from dotenv import load_dotenv
from fastapi import FastAPI
import time

app = FastAPI()

@app.get("/1")
async def awesome_endpoint():
    print("Hello AI PlayGrounds!")
    time.sleep(5)
    print("Bye AI PlayGrounds!")

@app.get("/2")
async def awesome_endpoint2():
    print("Hello Looper AI!")
    await asyncio.sleep(5)
    print("Bye Looper AI!")

@app.get("/3")
def awesome_endpoint3():
    print("Hello SpinOut!")
    time.sleep(5)
    print("Bye SpinOut!")
