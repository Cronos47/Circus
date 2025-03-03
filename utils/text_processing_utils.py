import json

from fix_busted_json import repair_json


def format_prompt(prompt):
    """Utility to preprocess prompts for providing better contextual understanding for the model"""

    prompt = prompt.split(" ")
    prompt = [word for word in prompt if len(word) > 0]
    prompt = " ".join(prompt)
    return prompt


def fix_faulty_json_string(response_text):
    """Utility to modify faulty json outputs"""

    response_text = " ".join([word for word in response_text.split(" ") if len(word) > 0])
    response_text = " ".join([word for word in response_text.split("\n") if len(word) > 0])
    response_text = response_text.replace("\\", "")

    if "{" not in response_text:
        if '"system":' not in response_text:
            response_text = '{"system": ' + response_text

        else:
            index = response_text.index('"system":')
            response_text = "{" + response_text[index:]

    if "}" not in response_text:
        response_text = response_text + "}"

    if '"' not in response_text.split(":")[1][:3]:
        response_text = response_text.split(":")[0] + \
                        ': "' + response_text.split(":")[1].split(" " + \
                                                                  response_text.split(":")[1])[-1]

    if ''.join(response_text.split(" ")).split("}")[0][-1] != '"':
        response_text = response_text.split("}")[0] + '"' + "}"

    if response_text.count('"') > 4:
        sub_segment = ' '.join(response_text.split(":")[1].split('"'))
        sub_segment = '"' + sub_segment[:sub_segment.find("}")] + '"' + "}"
        response_text = response_text.split(":")[0] + sub_segment
    return response_text


def format_json_style(response_text, prev_messages):
    """Utility to format deepseek response in json format"""

    response_text = fix_faulty_json_string(response_text)
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    json_str = response_text[json_start:json_end]

    parsed_json = json.loads(repair_json(str(json_str)))

    if len(prev_messages) != 0:
        prev_messages.append(parsed_json)
        return prev_messages.copy()

    else:
        return parsed_json["system"]


def format_message_to_role_mapper(messages, role, text):
    """Utility to accumulate roles and their respective contents"""

    messages.append({"role" : role, "content" : text})
    return messages.copy()


def format_message_to_whole_strings(messages):
    """Utility to format openAI prompt structure into whole strings"""

    formatted_prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted_prompt += f"{role}: {content}\n"
    return formatted_prompt
