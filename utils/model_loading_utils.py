import torch

# import openai
from openai import OpenAI

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .inference_utils import infer_openai_llms, infer_hf_transformers


def load_gpt(api_key, model_name,  system_prompt):
    """Utility to load gpt model 4o-mini"""

    client = OpenAI(api_key=api_key)
    client, system_prompt = infer_openai_llms(client, model_name,
                                              system_prompt, True)
    return client, system_prompt


def load_google_gemini(api_key, model_name, system_prompt):
    """Utility to load google gemini"""

    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=api_key)

    client, system_prompt = infer_openai_llms(client, model_name, 
                                              system_prompt, True)
    return client, system_prompt


def load_deepseek(model_name, system_prompt):
    """Utility to load deepseek-R1 model"""

    # Check for GPU availability, fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use FP16 instead of FP8 to avoid the GPU-only issue
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                                                device_map="auto") # Automatically maps to CPU or GPU
    # Set up text generation pipeline
    text_generator = pipeline("text-generation",
                                model=model,
                                tokenizer=tokenizer)

    text_generator, system_prompt = infer_hf_transformers(text_generator, system_prompt)
    return text_generator, system_prompt