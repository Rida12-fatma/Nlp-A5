import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_response(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

st.title("Direct Preference Optimization Model")

st.write("This app generates responses using a fine-tuned GPT-2 model.")

tokenizer, model = load_model()

prompt = st.text_area("Enter your prompt:")
if st.button("Generate Response"):
    if prompt:
        response = generate_response(prompt, tokenizer, model)
        st.write("### Response:")
        st.write(response)
    else:
        st.warning("Please enter a prompt.")
