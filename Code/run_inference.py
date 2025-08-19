# run_inference.py
# Script to run inference on the finetuned model after each 10 epochs using predefined Japanese sentences

MODEL_PATH = "./lora_output"  # Update if your model is saved elsewhere
SENTENCES = [
    "私は昨日映画を見ました。",
    "この本はとても面白いです。",
    "明日は雨が降るでしょう。",
    # Add more Japanese sentences as needed
]

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    for idx, sentence in enumerate(SENTENCES, 1):
        prompt = f"Translate the following Japanese sentence to Vietnamese: {sentence}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                do_sample=False,
                early_stopping=True
            )
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"[{idx}] Japanese: {sentence}")
        print(f"    Vietnamese: {result.split('Answer:')[-1].strip()}")
        print()

if __name__ == "__main__":
    main()
