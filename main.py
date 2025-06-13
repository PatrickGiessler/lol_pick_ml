import uvicorn

from fastapi import FastAPI
from app.api import router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Champion ML Predictor")

app.include_router(router)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8111)