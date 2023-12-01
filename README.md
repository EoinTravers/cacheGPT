# cacheGPT

`cachegpt` is a lightweight package that provides a persistant cache 
for (some) calls to the OpenAI API, using the `diskcache` package.
It is primarily intended for use in interactive data science work,
where it can save you a considerable amount of time and money.

## Usage

### Completions

```python
import cachegpt
# Create client. Optional arguments (e.g. model) are passed to the OpenAI completions API
gpt = cachegpt.GPT(cache_dir='data/gpt_cache', model='gpt-3.5-turbo')

# First call
%time gpt("Hello world", system_prompt="Respond in French")
# CPU times: user 24.6 ms, sys: 22 ms, total: 46.6 ms
# Wall time: 650 ms
# 'Bonjour le monde!'

# Second call - uses cache
%time gpt("Hello world", system_prompt="Respond in French")
# CPU times: user 1.11 ms, sys: 696 µs, total: 1.81 ms
# Wall time: 1.2 ms
# 'Bonjour le monde!'
```

## Text Embeddings


```python
embeddings = cachegpt.Embeddings(cache_dir='data/embedding_cache')
inputs = ["apple", "banana", "Mexico"]
# Default output is a dataframe with a column for each input
embeddings(inputs)
#          apple    banana    Mexico
# 0     0.007811 -0.013927 -0.000323
# 1    -0.022955 -0.032886 -0.006600
# 2    -0.007400  0.007615 -0.001157
# 3    -0.027779 -0.016562 -0.024243
# 4    -0.004670 -0.005112 -0.007740
# ...        ...       ...       ...
# 1531  0.021954  0.025218  0.001689
# 1532 -0.012101 -0.017800 -0.001647
# 1533 -0.013581 -0.006548  0.002300
# 1534 -0.015678 -0.017102 -0.018005
# 1535  0.006102  0.001874 -0.027697

# [1536 rows x 3 columns]
```

## Auth

By default, GPT() and Embeddings() use `dotenv` to look for a `.env` file and read in `OPENAI_KEY` as an environment variable.
Alternatively, you can pass an API key directly (by setting `api_key=="<your-key>")`, or prompt the user for one (`auth=="prompt"`).