import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datasets import load_dataset

# Set up the title of the app
st.title("Text Generation and Dataset Explorer")

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Load the SHP dataset
@st.cache_data
def load_and_preprocess_dataset():
    dataset = load_dataset("stanfordnlp/SHP", split="train")
    df = pd.DataFrame(dataset)
    df["preferred_response"] = df["labels"].apply(lambda x: "human_ref_A" if x == 0 else "human_ref_B")
    df = df.rename(columns={"history": "prompt", "human_ref_A": "response_1", "human_ref_B": "response_2"})
    df = df[["prompt", "response_1", "response_2", "preferred_response"]]
    return df

df = load_and_preprocess_dataset()

# Function to generate text
def generate_text(prompt, max_length=100):
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

# Streamlit app layout
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose an option", ["Text Generation", "Dataset Explorer"])

if option == "Text Generation":
    st.header("Text Generation")
    prompt = st.text_area("Enter your prompt:", "Once upon a time")
    if st.button("Generate Text"):
        with st.spinner("Generating text..."):
            generated_text = generate_text(prompt)
            st.success("Generated Text:")
            st.write(generated_text)

elif option == "Dataset Explorer":
    st.header("Dataset Explorer")
    st.write("Here is a sample of the dataset:")
    sample_size = st.slider("Select sample size", 1, 10, 5)
    sample_df = df.sample(sample_size)
    st.write(sample_df)
