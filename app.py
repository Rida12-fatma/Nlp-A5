import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Hugging Face model path (update with your actual model)
MODEL_NAME = "your-hf-username/my-trained-model"

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.title("🤖 NLP Model Demo")
st.write("Enter some text and see the model's response!")

# User input
user_input = st.text_area("Enter text:", "")

if st.button("Generate Response"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Model Response:")
        st.write(response)
    else:
        st.warning("Please enter some text!")

