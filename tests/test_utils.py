import unittest
from unittest.mock import patch
import openai
import os
from cachegpt.utils import openai_client


class TestOpenAIClient(unittest.TestCase):
    def test_with_api_key_arg(self):
        # Test when `api_key` is provided as an argument
        api_key = "my-api-key"
        client = openai_client(api_key=api_key, auth="arg")
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(client.api_key, api_key)

    @patch("os.getenv", return_value="my-env-api-key")
    def test_with_dotenv(self, mock_getenv):
        # Test when using 'dotenv' authentication method
        client = openai_client(auth="dotenv", env_var="OPENAI_KEY")
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(client.api_key, "my-env-api-key")
        mock_getenv.assert_called_once_with("OPENAI_KEY")

    @patch("getpass.getpass", return_value="my-prompt-api-key")
    def test_with_prompt(self, mock_getpass):
        # Test when using 'prompt' authentication method
        client = openai_client(auth="prompt")
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(client.api_key, "my-prompt-api-key")
        mock_getpass.assert_called_once()

    @patch("os.getenv", return_value=None)
    def test_missing_api_key(self, mock_getenv):
        # Test when `api_key` is not provided, and no authentication method is specified
        os.unsetenv("OPENAI_KEY")
        with self.assertRaises(ValueError) as context:
            openai_client(auth="dotenv", env_var="OPENAI_KEY")
        self.assertEqual(
            str(context.exception), "Environment variable `OPENAI_KEY` is not set"
        )
        mock_getenv.assert_called_once_with("OPENAI_KEY")

    def test_invalid_auth_method(self):
        # Test when an invalid authentication method is provided
        with self.assertRaises(NotImplementedError) as context:
            openai_client(auth="invalid_auth")
        self.assertEqual(
            str(context.exception),
            "`auth` must be one of `['dotenv', 'prompt', 'arg']`",
        )


if __name__ == "__main__":
    unittest.main()
