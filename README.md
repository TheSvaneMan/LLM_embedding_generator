# 1, Create a virtual environment using python to work from

# 2. Activate the virtual environment. In the terminal type the following command
# source llmapp/bin/activate

# 3. Install dependencies

# Chromadb
# https://docs.trychroma.com/
# Chroma is the open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.
pip install chromadb

# Langchain
# https://python.langchain.com/en/latest/getting_started/getting_started.html
# LangChain is a framework for developing applications powered by language models. 
pip install langchain

# LlamaIndex
# https://gpt-index.readthedocs.io/en/latest/index.html
# https://gpt-index.readthedocs.io/en/latest/getting_started/installation.html

pip install llama-index

# install Dot ENV
# https://pypi.org/project/python-dotenv/
pip install python-dotenv

# Install PDF interpreter module
pip install pypdf

# install pytorch
# https://pytorch.org/get-started/locally/#linux-pip
# !! IMPORTANT: Make sure to install torch based on your system requirements
pip3 install torch torchvision torchaudio

# Install transformers (robots in disguise)
pip install transformers

## -------- Model hyperlinks --------- ##
# https://huggingface.co/facebook/opt-iml-max-30b 
# 60 gb model, longer processing time, more parameters

# https://git-lfs.com/
# Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server like GitHub.com or GitHub Enterprise.

# install lfs if you don't have it already using the link below
# https://packagecloud.io/github/git-lfs/install

# --
git lfs install