import torch 

# import openai
from openai import OpenAI

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from text_processing_utils import format_message_to_whole_strings


def load_gpt(api_key, model_name,  system_prompt):
    """Utility to load gpt model 4o-mini"""

    client = OpenAI(api_key=api_key)
    chat = client.chat.completions.create(model=model_name, messages=system_prompt)
    reply = chat.choices[0].message.content
    print(reply)
    system_prompt.append({"role" : "system", "content" : reply})
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
    # Generate response
    output = text_generator(format_message_to_whole_strings(system_prompt), 
                            max_length=500,
                            temperature=0.7,
                            truncation=True)

    reply = output[0]['generated_text']
    system_prompt.append({"role" : "system", "content" : reply})
    return text_generator, system_prompt