An attempt at running a local instance of various LLMs using langchain, to interpret local documents to query Q&As. 

# Summary: 
Due to my system being capped at 8Gb GPU memory, inference could not compile on the larger models. 
The smaller LLM models do not exhibit reliable enough results to continue usage. 

# Learnings:
  Improved python programming.
  Learned about LLMs and vectorizing text data.
  Learned about GPU memory allocation.

An updated repo will be made in the future to further explore possibilities. 
If you have issues running the repo, the code is configured to work with my VM and the public repo is not tailored for mass sharing. 

# Issues you may arrive at are:
  Incompatible dependency versions. 
  Missing and/or incompatible GPU drivers 

# Warning! Continue at your own risk

I would consider LLMs to be experimental technology and only use code from trusted sources. Any harm or damage to property when using open-source technology is on the end-user. Take note.

<hr />

1. Create a virtual environment using python (anaconda) to work from

2. Activate the virtual environment. In the terminal type the following command
`source llmapp/bin/activate`

3. Install dependencies

# Install dependencies

Chromadb
https://docs.trychroma.com/
"Chroma is the open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs."
`pip install chromadb`

Langchain
https://python.langchain.com/en/latest/getting_started/getting_started.html
"LangChain is a framework for developing applications powered by language models."
`pip install langchain`

LlamaIndex
https://gpt-index.readthedocs.io/en/latest/index.html
https://gpt-index.readthedocs.io/en/latest/getting_started/installation.html

`pip install llama-index`

install Dot ENV
https://pypi.org/project/python-dotenv/
`pip install python-dotenv`

Install PDF interpreter module
`pip install pypdf`

install pytorch
https://pytorch.org/get-started/locally/#linux-pip
# !! IMPORTANT: Make sure to install torch based on your system requirements
`pip3 install torch torchvision torchaudio`

Install transformers (robots in disguise)
`pip install transformers`

## -------- Model hyperlinks --------- ##
https://huggingface.co/facebook/opt-iml-max-30b 
"60 gb model, longer processing time, more parameters"

https://git-lfs.com/
"Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server like GitHub.com or GitHub Enterprise."

install lfs if you don't have it already using the link below
https://packagecloud.io/github/git-lfs/install

`git lfs install`
