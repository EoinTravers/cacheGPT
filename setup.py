from setuptools import setup

setup(
    name="cachegpt",
    version="0.1.0",
    description="Simple persistant caching for OpenAI chats and embeddings",
    url="",
    author="Eoin Travers",
    author_email="eoin.travers@gmail.com",
    license="MIT",
    packages=["cachegpt"],
    install_requires=[
        "diskcache>=5.6.3",
        "numpy>=1.26.2",
        "openai>=1.3.0",
        "pandas>=2.1.3",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
)
