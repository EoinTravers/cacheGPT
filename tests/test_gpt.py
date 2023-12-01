import unittest
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock
from cachegpt import GPT, Embeddings
from dotenv import load_dotenv
import openai


class TestGPT(unittest.TestCase):
    def setUp(self):
        self.temp_cache_dir = tempfile.mkdtemp()
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_KEY")
        self.gpt = GPT(cache_dir=self.temp_cache_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_cache_dir)

    def test_gpt_response_caching(self):
        user_input = "Translate the following English text to French: 'Hello, world!'"
        response = self.gpt(user_input)

        # # Verify that the response is cached
        # self.assertIn((user_input, "You are a helpful assistant", '{}'), self.gpt.cache)

        # Call GPT again with the same input, it should return the cached response
        response_cached = self.gpt(user_input)
        self.assertEqual(response, response_cached)

    def test_gpt_response_format(self):
        user_input = "Translate the following English text to French: 'Hello, world!'"
        response = self.gpt(user_input)

        # Verify that the response is a string
        self.assertIsInstance(response, str)

    # def test_gpt_custom_prompt(self):
    #     user_input = "Translate the following English text to Spanish: 'Good morning!'"
    #     custom_prompt = "You are a language expert"
    #     response = self.gpt(user_input, system_prompt=custom_prompt)

    #     # Verify that the custom system prompt is used
    #     self.assertIn(custom_prompt, response)


if __name__ == "__main__":
    unittest.main()
