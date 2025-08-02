import functools
import re
import threading
import requests

import httpx as _httpx
import ida_kernwin
import openai
from pyexpat.errors import messages

from gepetto.models.base import LanguageModel
import gepetto.models.model_manager
import gepetto.config

_ = gepetto.config._

class LiteLLM(LanguageModel):
    @staticmethod
    def get_server() -> str:
        return gepetto.config.get_config("LiteLLM", "SERVER", "LITELLM_SERVER")

    @staticmethod
    def get_menu_name() -> str:
        return "OpenAI"

    @staticmethod
    def supported_models():
        try:
            link = LiteLLM.get_server()
            if not link:
                return []
            response = requests.get(f"{link}/models")
            response.raise_for_status()
            return [
                model["id"]
                for model in response.json().get("data", [])
                if model.get("object") == "model"
            ]
        except requests.RequestException:
            return []

    @staticmethod
    def is_configured_properly() -> bool:
        # The plugin is configured properly if the API key is provided, otherwise it should not be shown.
        return len(LiteLLM.supported_models()) > 0

    def __init__(self, model):
        self.model = model

        self.client = openai.OpenAI(
            api_key="",
            base_url=LiteLLM.get_server()
        )

    def __str__(self):
        return self.model

    def query_model(self, query, cb, stream=False, additional_model_options=None):
        """
        Function which sends a query to a GPT-API-compatible model and calls a callback when the response is available.
        Blocks until the response is received
        :param query: The request to send to the model. It can be a single string, or a sequence of messages in a
        dictionary for a whole conversation.
        :param cb: The function to which the response will be passed to.
        :param additional_model_options: Additional parameters used when creating the model object. Typically, for
        OpenAI, response_format={"type": "json_object"}.
        """
        if additional_model_options is None:
            additional_model_options = {}
        try:
            if type(query) is str:
                conversation = [
                    {"role": "user", "content": query}
                ]
            else:
                conversation = query

            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                stream=stream,
                **additional_model_options
            )
            if not stream:
                ida_kernwin.execute_sync(functools.partial(cb, response=response.choices[0].message.content),
                                         ida_kernwin.MFF_WRITE)
            else:
                for chunk in response:
                    delta = chunk.choices[0].delta
                    finished = chunk.choices[0].finish_reason
                    content = delta.content if hasattr(delta, "content") else ""
                    cb(content, finished)

        except openai.BadRequestError as e:
            # Context length exceeded. Determine the max number of tokens we can ask for and retry.
            m = re.search(r'maximum context length is \d+ tokens, however you requested \d+ tokens', str(e))
            if m:
                print(_("Unfortunately, this function is too big to be analyzed with the model's current API limits."))
            else:
                print(_("General exception encountered while running the query: {error}").format(error=str(e)))
        except openai.OpenAIError as e:
            print(_("{model} could not complete the request: {error}").format(model=self.model, error=str(e)))
        except Exception as e:
            print(_("General exception encountered while running the query: {error}").format(error=str(e)))

    # -----------------------------------------------------------------------------

    def query_model_async(self, query, cb, stream=False, additional_model_options=None):
        """
        Function which sends a query to {model} and calls a callback when the response is available.
        :param query: The request to send to {model}
        :param cb: Tu function to which the response will be passed to.
        :param additional_model_options: Additional parameters used when creating the model object. Typically, for
        OpenAI, response_format={"type": "json_object"}.
        """
        t = threading.Thread(target=self.query_model, args=[query, cb, stream, additional_model_options])
        t.start()

gepetto.models.model_manager.register_model(LiteLLM)
