import os
import string

from llama_cpp import Llama


class LLM:
    def __init__(self) -> None:
        self.llama = Llama(
            model_path=os.getenv("LLM_MODEL_PATH"),
            n_gpu_layers=-1,
            # seed=1337,
            n_ctx=2048,
        )

    def complete(self, query: str):
        output = self.llama(f"Q: {query}\nA: ", max_tokens=256, stop=["Q:", "\n"])
        return {
            "output": output["choices"][0]["text"],
        }

    def demo(self, query: str):
        # process text so that during the demo, any punctuation / spacing / capitalisation is ignored
        # might just change the test to be keyword based? to allow alternate phrasing?
        query = (
            query.replace(" ", "")
            .translate(str.maketrans("", "", string.punctuation))
            .casefold()
        )
        if query == "case 1":
            return "case 1 response"
        # and so on


def llm_response(query: str, demo: bool = True):
    """
    Queries the LLM given data and returns the relevant response.

    Args:
        query (str): The input given by the user to the LLM.
        demo (bool): Whether to use pre-determined responses for the demo. Default True
    """
    # query the dmas database to get the information regardless of real or fake
    # for demo, for each case of 3 situations
    # depending on the values spit out the predetermined response
    # for not demo prototype thing, use marks' thingy
    return {"llm_response", "placeholder"}
