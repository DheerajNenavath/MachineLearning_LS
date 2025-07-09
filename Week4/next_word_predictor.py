from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token 

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def predict_next_words(prompt_text, max_new_tokens=10):
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Prompt:", prompt_text)
    print("Prediction:", generated_text)

while True:
    prompt = input("\nEnter a prompt (or type 'exit'): ")
    if prompt.lower() == 'exit':
        break
    predict_next_words(prompt, max_new_tokens=10)

