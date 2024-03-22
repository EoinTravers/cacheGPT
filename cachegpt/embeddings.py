import json
from typing import Callable, Union, Literal, Iterable
import openai
import diskcache
import numpy as np
import pandas as pd
from .utils import openai_client


class Embeddings:
    """Get text embeddings from OpenAI. Responses are cached, indefinitely, by default.

    Parameters:
        cache_dir (str, optional): The directory where caching data should be stored.
                                   Defaults to 'embedding_cache'.
        cache_timeout (float, optional): The maximum time in seconds for which cached embeddings are considered valid.
                                         Defaults to positive infinity.
        format (str, optional): The output format.
                                Options:
                                - 'df', 'df_c': Pandas DataFrame with one column per input text
                                - 'df_r':  Pandas DataFrame with one row per input text
                                - 'array': Numpy array of shape (n_inputs, embedding_dim)
                                - 'list': Simple list of lists

    Attributes:
        cache (diskcache.Cache): A caching object that stores embeddings.

    Methods:
        __call__(self, inputs: list[str], format: str = 'df') -> Union[pd.DataFrame, list, np.ndarray]:
            Generate embeddings for a list of text inputs and return them in the specified format.

    Example:`
        # Create an instance of the Embeddings class
        embeddings = Embeddings(cache_dir='custom_cache', cache_timeout=3600, format='df')

        # Generate embeddings for a list of text inputs
        input_texts = ["apple", "banana", "cherry"]
        embedding_df = embeddings(input_texts, format='df')
        print(embedding_df)
    """

    def __init__(
        self,
        api_key: Union[None, str] = None,
        auth: Literal["dotenv", "prompt", "arg"] = "dotenv",
        env_var: str = "OPENAI_KEY",
        cache_dir: Union[str, None] = "embedding_cache",
        cache_timeout: float = float("inf"),
        format="df",
        **embedding_args,
    ):
        if cache_dir is None:
            self.cache = None
        else:
            self.cache = diskcache.Cache(cache_dir, timeout=cache_timeout)
        self.client = openai_client(api_key, auth, env_var)
        default_args = {"model": "text-embedding-3-large", "dimensions": 1024}
        # Use defaults unless provided
        self.default_embedding_args = default_args | embedding_args

    def __call__(
        self,
        inputs: list[str],
        format="df",
        **embedding_args,
    ) -> Union[pd.DataFrame, list, np.ndarray]:
        # Validate inputs
        allowed_fmts = ["df", "df_c", "df_r", "array", "list"]
        if not format in allowed_fmts:
            raise ValueError(f"Format must be one of {allowed_fmts}, not '{format}")
        embeddings = []
        assert isinstance(inputs, Iterable) and not isinstance(inputs, str)
        # inputs = [str(x) for x in inputs]
        kwargs = self.default_embedding_args | embedding_args
        for inp in inputs:
            cache_key = (inp, json.dumps(kwargs, sort_keys=True))
            if self.cache is not None and cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                response = self.client.embeddings.create(input=[inp], **kwargs)
                x = response.data[0].embedding
                if self.cache is not None:
                    self.cache[cache_key] = x
                embeddings.append(x)
        if format == "df" or format == "df_c":
            return pd.DataFrame(embeddings, index=inputs).T
        elif format == "df_r":
            return pd.DataFrame(embeddings, index=inputs)
        elif format == "list":
            return embeddings
        elif format == "array":
            return np.array(embeddings)
