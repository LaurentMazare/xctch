import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
with open("scripts/llama3/vocab.json", "w") as fobj:
    json.dump(tokenizer.vocab, fobj)
