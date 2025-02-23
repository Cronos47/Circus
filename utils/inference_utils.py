from .text_processing_utils import format_message_to_whole_strings, format_json_style


def decide_contest_result(score1, score2):
    """Utility to decide the winner of the battle based on scores obtained"""

    if score1 > score2:
        print("GPT-4 is the WINNER !!")

    elif score1 == score2:
        print("AHH CRAP THIS GAME IS A DRAW!")

    else:
        print("GEMINI is the WINNER!!")


def infer_openai_llms(client, model_name, messages, load_phase=False):
    """Utility to infer on openai models"""

    chat = client.chat.completions.create(model=model_name,
                                          messages=messages)

    if not load_phase:
        reply = format_json_style(chat.choices[0].message.content, [])

    else:
        reply = chat.choices[0].message.content

    messages.append({"role" : "system", "content" : reply})
    return client, messages


def infer_hf_transformers(text_generator, messages):
    """Utility to infer on huggingface transformers"""

    output = text_generator(format_message_to_whole_strings(messages),
                            max_length=500,
                            temperature=0.7,
                            truncation=True)

    reply = output[0]['generated_text']
    messages.append({"role" : "system", "content" : reply})
    return text_generator, messages
