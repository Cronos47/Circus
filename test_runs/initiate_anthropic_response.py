import os

import anthropic


# Set your API key
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

# Generate a response
response = client.messages.create(
    model="claude-2",  # Or use "claude-1" or other versions
    max_tokens=300,
    messages=[{"role": "user", "content": "Generate a rap to defeat GPT-4"}]
)

print(response.content)
