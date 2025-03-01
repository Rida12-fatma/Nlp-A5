https://nlp-a5-kp8tfkod6jhbhysxccimyz.streamlit.app/


# My Trained Model

This repository contains my fine-tuned model, which has been trained using Direct Preference Optimization (DPO). The model is now publicly available on the Hugging Face Model Hub.

## 🔗 Model Link
You can access and use the model here:  
[**My Trained Model on Hugging Face**](https://huggingface.co/R1243/my-trained-model/tree/main)

## 📂 Files Included
- `pytorch_model.bin` - The trained model weights.
- `config.json` - Configuration file for the model.
- `tokenizer.json` - Tokenizer used during training.
- `tokenizer_config.json` - Tokenizer configuration.

## 🚀 Usage
You can load the model using the `transformers` library as follows:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your-hf-username/my-trained-model"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example usage
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 📢 Citation
If you use this model in your work, please consider citing it:
```bibtex
@misc{your_model,
  author = {Your Name},
  title = {My Trained Model},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/your-hf-username/my-trained-model}
}
```

## 🤝 Contributing
Feel free to open an issue or submit a pull request if you have improvements or suggestions!

## 📜 License
This model is shared under the **MIT License** (or specify the appropriate license).

