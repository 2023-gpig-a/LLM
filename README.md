# LLM - Large Language Model

This should process queries from the user about the data

## Setup
Use [Poetry](https://python-poetry.org/docs/) to install all relevant packages. Run `poetry install`#

[Download the Gemma model](https://huggingface.co/google/gemma-1.1-2b-it-GGUF) and put the path to it in .env

## Running without docker

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn llm.app.main:app --reload
```

## Running with docker
```
docker build -t llm_image .
docker run -d --name llm -p 8081:8081 llm_image
```

## Endpoints:

`/query_llm`
When a post request is made to this endpoint, the LLM will process the given query from the user and return a suggested course of action based on the question and the plant growth data from the database (? actual use cases are dependent on findings from interviews)

Returns: `{"llm_response": llm_response}`
