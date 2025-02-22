""" This module aims to make a wrapper for Llama """

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login


class LLamaWrapper:
    """
    This is a wrapper for Llama model
    """
    def __init__(self, model_name, hf_token=None, cache_dir=None, use_local_only=False):
        """
        Initializes the LLamaWrapper class.

        Args:
            model_name (str): Hugging Face model ID (e.g., 'meta-llama/Llama-2-7b-hf').
            hf_token (str): Hugging Face Access Token for gated models.
            cache_dir (str): Directory to cache model files.
            use_local_only (bool): If True, only uses local cached files (no redownloading).
        """

        self.model_name = model_name
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self.use_local_only = use_local_only
        self.model = None
        self.tokenizer = None
        self.generator = None

        # Authenticate with Hugging Face if token is provided
        if self.hf_token:
            login(self.hf_token)

        # Load tokenizer and model
        self._load_model()

    def _load_model(self):
        """Loads the tokenizer and model, using cache or local files if available."""

        print(f"Loading model: {self.model_name}")

        # Set cache directory if provided
        cache_kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_token,
            local_files_only=self.use_local_only,
            **cache_kwargs
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_token,
            local_files_only=self.use_local_only,
            **cache_kwargs
        )

        # Setup pipeline
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        print("Model loaded successfully.")


    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """
        Generates text from the model.

        Args:
            prompt (str): The input prompt for text generation.
            max_length (int): Max tokens to generate.
            temperature (float): Sampling temperature (higher = more creative).
            top_p (float): Nucleus sampling parameter.

        Returns:
            str: Generated text.
        """

        print(f"Generating text for prompt: {prompt}")
        outputs = self.generator(prompt, 
                                 max_length=max_length, 
                                 temperature=temperature, 
                                 top_p=top_p)
        return outputs[0]['generated_text']


    def save_local(self, save_path):
        """
        Saves the tokenizer and model locally for future use.

        Args:
            save_path (str): Directory path to save model files.
        """
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")


    def load_local(self, local_path):
        """
        Loads the model and tokenizer from a local directory.

        Args:
            local_path (str): Path to the locally saved model.
        """
        print(f"Loading model from local path: {local_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.model = AutoModelForCausalLM.from_pretrained(local_path)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        print("Local model loaded successfully.")
