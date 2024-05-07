from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model.main import llm_response

import dotenv

dotenv.load_dotenv()  # this loads the environment variable for the model path


class Query(BaseModel):
    query: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # frontend dev
        "http://localhost:8080",  # frontend Docker
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def hello_world():
    return {"response": "Hello World"}


@app.post("/query_llm")
async def query_llm(query: Query):
    return llm_response(query)
