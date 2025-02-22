import os

from llama_wrapper import LLamaWrapper


model_name = "meta-llama/Llama-2-7b-hf"
hf_token = os.getenv("HUGGING_FACE_TOKEN")
llama_model_path = "./local_llama2_model"

# Initialize LLamaWrapper with Hugging Face token
llama = LLamaWrapper(model_name=model_name, hf_token=hf_token)

# Save the downloaded model locally
llama.save_local(llama_model_path)

# Load from local storage in the future (no redownloading)
llama_local = LLamaWrapper(model_name=llama_model_path, use_local_only=True)

# Generate text again
output = llama_local.generate_text("What is quantum computing?", max_length=100)
print("Generated Text:\n", output)
