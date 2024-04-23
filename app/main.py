import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import dotenv

dotenv.load_dotenv()

llama = Llama(
    model_path=os.getenv('LLM_MODEL_PATH'),
    n_gpu_layers=-1,
    # seed=1337,
    n_ctx=2048,
)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", # frontend dev
        "http://localhost:8080", # frontend Docker
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/complete")
async def complete(query: str):
    output = llama(
        f'Q: {query}\nA: ',
        max_tokens=256,
        stop=['Q:', '\n']
    )
    return {
        'output': output['choices'][0]['text'],
    }
