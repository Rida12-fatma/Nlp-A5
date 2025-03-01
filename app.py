import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from datasets import load_dataset, Dataset
import pandas as pd

def load_and_preprocess_data():
    dataset = load_dataset("stanfordnlp/SHP", split="train")
    df = pd.DataFrame(dataset)
    df["preferred_response"] = df["labels"].apply(lambda x: "human_ref_A" if x == 0 else "human_ref_B")
    df = df.rename(columns={"history": "prompt", "human_ref_A": "response_1", "human_ref_B": "response_2"})
    df = df[["prompt", "response_1", "response_2", "preferred_response"]]
    return Dataset.from_pandas(df)

def load_model(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def main():
    dataset = load_and_preprocess_data()
    model, tokenizer = load_model()
    print("Model and dataset loaded successfully!")

if __name__ == "__main__":
    main()
