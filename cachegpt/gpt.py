import json
from typing import Callable, Literal
import openai
import diskcache
import pandas as pd
from .utils import openai_client


class GPT:
    """Send messages to OpenAI and get a response. Responses are cached, indefinitely, by default.

    Parameters:
        cache_dir (str, optional): The directory where caching data should be stored. Defaults to 'gpt_cache'.
        cache_timeout (float, optional): The maximum time in seconds for which cached responses are considered valid.
            Defaults to positive infinity. Be careful with this.
        **gpt_args: Additional keyword arguments for configuring the GPT model.
            These override the default GPT settings.

    Attributes:
        cache (diskcache.Cache): A caching object that stores GPT responses.
        default_gpt_args (dict): Default configuration settings for the GPT model.
            The following are included:
            - 'model': 'gpt-3.5-turbo'
            - 'temperature': 0
            - 'frequency_penalty': 0
            - 'max_tokens': 200
            - 'n': 1
            - 'presence_penalty': 0
            - 'response_format': {'type': 'text'}
            - 'seed': 1234
            - 'top_p': 1

    Methods:
        __call__(self, user_input: str, system_prompt: str = "You are a helpful assistant", **gpt_args) -> str:
            Sends a user input message to the GPT model and returns the generated response.

    Examples:
        # Create a GPT instance with custom caching and GPT configuration
        gpt = GPT(cache_dir='gpt_cache', model='gpt-4')
        # Send a user input to GPT and get a response
        gpt("Translate the following English text to French: 'Hello, world!'")
        # 'Bonjour, monde !'
    """

    def __init__(
        self,
        api_key: None | str = None,
        auth: Literal["dotenv", "prompt", "arg"] = "dotenv",
        env_var: str = "OPENAI_KEY",
        cache_dir: str = "gpt_cache",
        cache_timeout: float = float("inf"),
        **gpt_args
    ):
        self.cache = diskcache.Cache(cache_dir, timeout=cache_timeout)
        self.client = openai_client(api_key, auth, env_var)
        default_args = {
            "model": "gpt-3.5-turbo",
            "temperature": 0,
            # The rest are all actual OpenAI defaults, just included here to simplify caching
            "frequency_penalty": 0,
            "max_tokens": 200,
            "n": 1,
            "presence_penalty": 0,
            "response_format": {"type": "text"},
            "seed": 1234,
            "top_p": 1,
        }
        # Use defaults unless provided
        self.default_gpt_args = default_args | gpt_args

    def __call__(
        self,
        user_input: str,
        system_prompt: str = "You are a helpful assistant",
        **gpt_args
    ) -> str:
        # TODO: Set these at class level
        kwargs = self.default_gpt_args | gpt_args
        cache_key = (user_input, system_prompt, json.dumps(kwargs, sort_keys=True))
        if self.cache is not None and cache_key in self.cache:
            return self.cache[cache_key]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
            response = self.client.chat.completions.create(messages=messages, **kwargs)
            result = response.choices[0].message.content
            if self.cache is not None:
                self.cache[cache_key] = result
            return result
