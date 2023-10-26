from fastapi import FastAPI
from llm import run
import logging, uvicorn, os

app = FastAPI()

@app.get('/')
async def health_check():
    return 200

@app.get('/query')
async def query(q: str = ''):
    return run(q)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=os.getenv('PORT', 8080))
    logging.info('Server started')