# In your changeupstuff.py, ensure the token is passed correctly
# Example of how to load it explicitly:
import os
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
token = "your_hf_token_here" # Or load from os.environ.get("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
