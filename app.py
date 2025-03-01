from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

# Load the SHP dataset
def load_and_preprocess_dataset():
    dataset = load_dataset("stanfordnlp/SHP", split="train")
    
    def preprocess_dataset(dataset):
        df = pd.DataFrame(dataset)
        df["preferred_response"] = df["labels"].apply(lambda x: "human_ref_A" if x == 0 else "human_ref_B")
        df = df.rename(columns={"history": "prompt", "human_ref_A": "response_1", "human_ref_B": "response_2"})
        df = df[["prompt", "response_1", "response_2", "preferred_response"]]
        return df
    
    df = preprocess_dataset(dataset)
    return df

# Load model and tokenizer
def load_model_and_tokenizer(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Function to generate text
def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    df = load_and_preprocess_dataset()
    print("Sample preprocessed data:")
    print(df.head())

    model, tokenizer = load_model_and_tokenizer()

    prompt = "Once upon a time"
    generated_text = generate_text(model, tokenizer, prompt)
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
