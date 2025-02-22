import os
import json

import openai
from openai import OpenAI


def format_json_style(response_text, prev_messages):
    """Utility to format deepseek response in json format"""

    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    json_str = response_text[json_start:json_end]
    parsed_json = json.loads(json_str)
    prev_messages.append(parsed_json)
    return prev_messages


openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#pylint: disable=fixme
# TODO: The 'openai.my_api_key' option isn't read in the client API.
# You will need to pass it when you instantiate the client,
# e.g. 'OpenAI(my_api_key=os.getenv("OPENAI_API_KEY"))' or
# openai.my_api_key = os.getenv("OPENAI_API_KEY")

messages = [{"role": "system",
            "content": "You are an intelligent assistant who can rap. Generate only the rap only in a json format\
            where there will be a single key named 'system' and the value will be the rap itself"}]

message = "User : it seems DeepSeek wants to challenge you in a rap battle you up for it? Dont generate rap now"
if message:
    messages.append(
        {"role": "user", "content": message},
    )

print(messages)
print()

chat = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
reply = chat.choices[0].message.content

message = "This is what you rival had to say.\n Deepseek: rap1, rap2, rap3, rap4\n GPT cant compete with me no more!" + \
            "\nWhats your reply?"
if message:
    messages.append(
        {"role": "user", "content": message},
    )

chat = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
reply = chat.choices[0].message.content

print(f"ChatGPT: {reply}")

messages = format_json_style(reply, messages)
print()
print(messages)
