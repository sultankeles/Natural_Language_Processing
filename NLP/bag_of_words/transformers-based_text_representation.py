from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Transformers can be used for natural language proccessing."

inputs = tokenizer(text, return_tensors = "pt")  # output is returned as a PyTorch tensor

with torch.no_grad():    # The calculation of gradients is stopped, thus allowing for more efficient memory use.
    outputs = model(**inputs)
    
last_hidden_state = outputs.last_hidden_state

first_token_embedding = last_hidden_state[0, 0, :].numpy()

print(f"Text Representation: {first_token_embedding}")
