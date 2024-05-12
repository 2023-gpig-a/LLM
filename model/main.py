import os
import string

import requests

from llama_cpp import Llama


class LLM:
    def __init__(self) -> None:
        self.llama = Llama(
            model_path=os.getenv("LLM_MODEL_PATH"),
            n_gpu_layers=-1,
            # seed=1337,
            n_ctx=2048,
        )

    def complete(self, context: str, query: str):
        output = self.llama(
            f"{context}\n Q: {query}\nA: ", max_tokens=256, stop=["Q:", "\n"]
        )
        return {
            "output": output["choices"][0]["text"],
        }


class DemoLLM:
    def __init__(self) -> None:
        self.context_data = {}
        self.growthtypes = {}

    def update_context_data(self, context_data: dict) -> None:
        """
        Update the "LLM" with the newest data from DMAS
        """
        self.context_data = context_data
        self.growthtypes = self._get_growth_types

    def _get_growth_types(self) -> dict:
        """
        returns growth types for each plant in the dummy data (mainly, for whether it is showing exponential growth)
        for the demo, there's 2 plants, so I hard-coded it as such
        (There's probably a more efficient way to do this, but it should work for now)
        """
        context_data = self.context_data
        growth_patterns = {}
        plantnames = context_data.keys()
        for plants in [plantnames, plantnames[::-1]]:
            if (
                context_data[plants[1]] == "GROWING"
                and context_data[plants[2]] == "DECAYING"
            ):
                growth_patterns[
                    plants[1]
                ] = "showing potential destructive exponential growth"
                growth_patterns[plants[2]] = "decaying"
            elif (
                context_data[plants[1]] == "GROWING"
                and context_data[plants[2]] == "CONSTANT"
            ):
                growth_patterns[
                    plants[1]
                ] = "showing non-destructive exponential growth"
                growth_patterns[plants[2]] = "showing normal growth"
            elif context_data[plants[1]] == "CONSTANT":
                growth_patterns[plants[1]] = "showing normal growth"
            elif context_data[plants[1]] == "DECAYING":
                growth_patterns[plants[1]] = "decaying"
        return growth_patterns

    def demo(self, query: str):
        """
        Processes text during the demo to spit out a "LLM response".
        Currently I've hardcoded it to work with a set list of dummy questions:
        1- [Growth patterns] What are the growth patterns of the plants in this area?
        2- [Destructive] Is the X plant showing destructive exponential growth?
        3- [Growth] What sort of growth is the X plant showing?
        4- [Explanation] What other possible explanations could there be for this behaviour?
        5- [Do] Can you give an example of something I could do to stop this?
        """
        # process text so that during the demo, any punctuation / spacing / capitalisation is ignored
        query = (
            query.replace(" ", "")
            .translate(str.maketrans("", "", string.punctuation))
            .casefold()
        )

        context_data = self.context_data
        growthtypes = self.growthtypes

        if "growthpatterns" in query:  # q1
            response = ""
            for key in context_data.keys():
                response += (
                    f"The {key} plant appears to be {context_data[key].casefold()}. "
                )
            return response

        elif "growth" in query:  # q2 and 3
            plant = "rose" if "rose" in query else "knotweed"
            plant_growthtype = growthtypes[plant]
            if "destructive" in query:  # q2
                if "potentially destructive" in plant_growthtype:
                    return f"The {plant} plant may potentially be showing destructive exponential growth."
                else:
                    return f"The {plant} plant does not appear to be showing destructive exponential growth."
            else:
                return f"The {plant} plant appears to be {plant_growthtype}."

        elif "explanation" in query:  # q4
            if "potentially destructive" in growthtypes.values():
                return "Another possible explanation for one plant to spread whilst others decay could be that an animal is eating other plants in the area."
            elif "non-destructive" in growthtypes.values():
                return "Another possible explanation for one plant to spread could be that a lot of this species has been planted recently."
            # Surely we're not going to ask about explanations for normal growth, right? Not sure what to say for that, so leaving it for now.

        elif "do" in query:  # q5
            # just going to write for the destructive case for now
            # https://www.gardenersworld.com/how-to/solve-problems/japanese-knotweed-removal/
            # https://www.gov.uk/guidance/prevent-japanese-knotweed-from-spreading
            return "To stop the spread of knotweed, a glyphosate-based weedkiller can be applied several times. The plants could also be buried deep enough to prevent them from regrowing."


class LLM_Manager:
    def __init__(self, demo: bool) -> None:
        """
        Creates either a real or a fake LLM instance.

        Args:
            demo (bool): Whether to use pre-determined responses for the demo, if true creates a fake LLM instance.
        """
        self.is_demo = demo
        if demo:
            self.llm_instance = DemoLLM()
        else:
            self.llm_instance = LLM()
        self.dmas_endpoint = (
            os.getenv("VITE_DMAS_ENDPOINT") + "/estimate_growth_pattern"
        )

    def llm_response(self, query: str = True):
        """
        Queries the LLM given data and returns the relevant response.

        Args:
            query (str): The input given by the user to the LLM.
        """
        # query the dmas database to get the information regardless of real or fake
        dmas_endpoint = self.dmas_endpoint
        context_data = requests.get(dmas_endpoint)  # this is a dict

        """Example growth pattern dictionary:
        {"Rose": "CONSTANT",
        "Knotweed: "DECAYING"}
        """

        # send the query and data to the LLM

        if self.is_demo:
            self.llm_instance.update_context_data(context_data)
            response = self.llm_instance.demo(
                context_data, query
            )  # easier to just use the dictionary for dummy function
        else:
            # process the context data into a context string
            context = (
                f"There are {len(context_data)} plants growing in an area in a forest."
            )
            for key in context_data.keys():
                context += (
                    f" The {key} plant is in a {context_data[key].casefold()} state."
                )
            response = self.llm_instance.complete(context, query)

        return {"llm_response": response}
