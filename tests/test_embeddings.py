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


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.temp_cache_dir = tempfile.mkdtemp()
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_KEY")
        self.embeddings = Embeddings(cache_dir=self.temp_cache_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_cache_dir)

    def test_embeddings_caching(self):
        input_texts = ["apple", "banana", "cherry"]
        embeddings = self.embeddings(input_texts)

        # Verify that the embeddings are cached
        for text in input_texts:
            self.assertIn(text, self.embeddings.cache)

        # Call embeddings again with the same inputs, they should return the cached values
        embeddings_cached = self.embeddings(input_texts)
        self.assertTrue(np.array_equal(embeddings, embeddings_cached))

    def test_embeddings_formats(self):
        input_texts = ["apple", "banana", "cherry"]

        # Test DataFrame format
        embedding_df = self.embeddings(input_texts, format="df")
        self.assertIsInstance(embedding_df, pd.DataFrame)
        self.assertEqual(len(embedding_df.columns), len(input_texts))

        # Test array format
        embedding_array = self.embeddings(input_texts, format="array")
        self.assertIsInstance(embedding_array, np.ndarray)
        self.assertEqual(embedding_array.shape, (len(input_texts), 1536))

        # Test list format
        embedding_list = self.embeddings(input_texts, format="list")
        self.assertIsInstance(embedding_list, list)
        self.assertEqual(len(embedding_list), len(input_texts))

    def test_embedding_distances(self):
        input_texts = ["apple", "banana", "Mexico"]
        embedding_df = self.embeddings(input_texts, format="df")
        r = embedding_df.corr()
        self.assertGreater(r.loc["apple", "banana"], r.loc["apple", "Mexico"])


if __name__ == "__main__":
    unittest.main()
