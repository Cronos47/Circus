import os

import openai

# Use your OpenRouter API key
openrouter_key = os.getenv("OPENROUTER_KEY")

client = openai.OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=openrouter_key)  # Directs the openai SDK to OpenRouter

generation_instruction = " Generate only the rap, in a proper json format \
                           where the json will have a single key named 'system' and \
                           the value of that key will be the rap itself."

# Same OpenAI-style API call
response = client.chat.completions.create(
    model="google/gemini-2.0-flash-lite-preview-02-05:free",  # Choose your preferred model
    messages=[
        {"role": "system", "content": "You are a helpful assistant who can rap." + generation_instruction},
        {"role": "user", "content": "Generate a rap to defeat GPT-4."}
    ],
    temperature=0.7,
    max_tokens=500
)

reply = response.choices[0].message.content

# Output the response
print(reply)
