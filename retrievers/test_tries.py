import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer 
import pickle
import torch
from GENRE.genre.trie import Trie

print("LOAD TRIE")
# load the prefix tree (trie)
with open("kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

print(trie.get(""))
device = 2
gen_len = 10
model_checkpoint = "gpt2"

print("LOAD MODEL")
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model.to(device)


print("GENERATE")

input_ids = tokenizer("I love cats", return_tensors='pt')
input_len = len(input_ids['input_ids'][0])
with torch.no_grad():
    output = model.generate(
        input_ids = input_ids['input_ids'].to(device),
        max_length=input_len+gen_len,
        num_beams=1,
        prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())
    )

response = tokenizer.decode(output[0][input_len:])
print(response)