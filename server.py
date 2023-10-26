from fastapi import FastAPI
from llm import run
import logging

app = FastAPI()

@app.get('/')
async def health_check():
    return 200

@app.get('/query')
async def query(q: str = ''):
    return run(q)

logging