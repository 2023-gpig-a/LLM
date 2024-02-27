from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from model.main import llm_response

class Query(BaseModel):
    query: str

app = FastAPI()

@app.get("/")
async def hello_world():
    return {"response": "Hello World"}


@app.post("/query_llm")
async def query_llm(query: Query):
    return llm_response(query)