import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"
hf_token = os.getenv("HUGGING_FACE_TOKEN")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)

# Load model with offloading (no bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,      # For CPU-only setups
    device_map="auto",              # Auto maps to CPU
    offload_folder="./offload",     # Offload large layers to disk
    use_auth_token=hf_token
)

# Prepare prompt
prompt = "Explain the concept of relativity in simple terms."
inputs = tokenizer(prompt, return_tensors="pt")

# Inference
outputs = model.generate(
    **inputs,
    max_length=200,
    temperature=0.7,
    top_p=0.9
)

# Print output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
