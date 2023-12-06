import os
from typing import Literal, Union
import openai


def openai_client(
    api_key: Union[None, str] = None,
    auth: Literal["dotenv", "prompt", "arg"] = "dotenv",
    env_var: str = "OPENAI_KEY",
) -> openai.OpenAI:
    """
    Initializes an OpenAI client object while handling different authentication methods.

    Parameters:
    - api_key (str or None): The OpenAI API key to authenticate the client. Onlyused if auth == "arg"
    - auth (str): The authentication method to use. Options are 'dotenv' (default), 'prompt', or 'arg'.
    - env_var (str): The environment variable to retrieve the API key when using 'dotenv' authentication.

    Returns:
    - openai.OpenAI: An instance of the OpenAI client with the provided API key.

    Raises:
    - ValueError: If the specified authentication method fails to retrieve an API key.
    - NotImplementedError: If an unsupported authentication method is provided.
    """
    if api_key is None:
        if auth == "dotenv":
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv(env_var)
            if api_key is None:
                raise ValueError(f"Environment variable `{env_var}` is not set")
        elif auth == "prompt":
            from getpass import getpass

            api_key = getpass("Enter your OpenAI API key: ")
        elif auth == "arg":
            raise ValueError("`api_key` argument must be provided if `auth == 'arg'")
        else:
            raise NotImplementedError(
                "`auth` must be one of `['dotenv', 'prompt', 'arg']`"
            )
    return openai.OpenAI(api_key=api_key)
