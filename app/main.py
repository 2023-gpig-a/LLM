from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model.main import LLM_Manager

import dotenv
import os

dotenv.load_dotenv()  # this loads the environment variable for the model path

llm_manager = LLM_Manager(demo=os.getenv("IS_DEMO"))


class Query(BaseModel):
    query: str


app = FastAPI(
    title="LLM",
    summary="Sends queries from the frontend to a large language model.",
    description="""This system interfaces with a Large Language Model using the LLaMA-CPP library.
    Plant growth data is pulled from DMAS and sent to a model alongside a query entered into the corresponding frontend page.
    This abstracts the workings of the LLM from the user and allows different models to be used to keep up with state-of-the-art.

    An endpoint is provided for the frontend systems to call the LLM and obtain its output. This is:
    -  `POST /query_llm`
""",
)

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


@app.get("/", summary="Hello world sanity check.")
async def hello_world():
    return {"response": "Hello World"}


@app.post(
    "/query_llm",
    summary="Send a query to the LLM and recieve a response from it.",
    description="Intended for frontend usage.",
)
async def query_llm(query: Query):
    return llm_manager.llm_response(query.query)
