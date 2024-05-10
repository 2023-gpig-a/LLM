import requests
import yaml

with open("./model/api_token.yml", "r") as file:
    token_file = yaml.safe_load(file)
API_TOKEN = token_file["API_TOKEN"]


def queryLLM(payload):
    API_URL = "https://api-inference.huggingface.co/models/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def llm_response(query: str = "Should Species A be classed as an invasive species?"):
    plantdata = [
        0.01,
        0.01,
        0.04,
        0.1,
        0.13,
        0.2,
    ]  # dummy data, this should be fetched from dmas
    dataquery = f"Each element in {plantdata} represents the proportion of plants in a forest identified as Species A each day. If the proportion of plants increases rapidly each day, a species is an invasive species."
    output = queryLLM({"inputs": {"question": query, "context": dataquery},})
    return {"llm_response": output["answer"]}


if __name__ == "__main__":
    print(llm_response()["llm_response"])
