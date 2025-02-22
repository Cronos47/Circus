import os

import numpy as np

from utils.model_loading_utils import load_gpt, load_deepseek
from utils.text_processing_utils import format_message_to_role_mapper, format_json_style
from utils.text_processing_utils import format_message_to_whole_strings
from utils.constants import LLAMA_MODEL_NAME, GPT_MODEL_NAME


def begin_circus(system_prompts, rounds):
    """Main function to build AI circus and decide winner of the battle!"""

    gpt, system_prompts[0] = load_gpt(os.getenv("OPENAI_API_KEY"), GPT_MODEL_NAME, system_prompts[0])
    deepseek, system_prompts[1] = load_deepseek(LLAMA_MODEL_NAME, system_prompts[1])
    judge, system_prompts[2] = load_gpt(os.getenv("OPENAI_API_KEY"), GPT_MODEL_NAME, system_prompts[2])

    toss_and_pass = np.random.randint(low=0, high=1)
    initial_prompt_part = "Let the rap battle begin! Whats your reply?"
    judge_prompt = "What would you rate these two raps out of 1 to 10? Return only the score\
                    delimited by a comma and nothing else.\n"

    gpt_message = deepseek_message = initial_prompt_part
    score_gpt = score_deepseek = 0

    for _ in range(rounds):
        if toss_and_pass == 0:
            system_prompts[0] = format_message_to_role_mapper(system_prompts[0],
                                                              role="user",
                                                              text=gpt_message)

            gpt_response = gpt.chat.completions.create(model=GPT_MODEL_NAME,
                                                       messages=system_prompts[0])

            gpt_reply = format_json_style(gpt_response.choices[0].message.content, [])
            formatted_gpt_reply = "This is what your rival had to say\n" + f"GPT : {gpt_reply}"

            system_prompts[0] = format_message_to_role_mapper(system_prompts[0], 
                                                              role="system",
                                                              text=gpt_reply)

            system_prompts[1] = format_message_to_role_mapper(system_prompts[1], 
                                                              role="user",
                                                              text=formatted_gpt_reply)

            deepseek_reply = deepseek(format_message_to_whole_strings(system_prompts[1]), 
                                       max_length=500,
                                       temperature=0.7)[0]["generated_text"]

            system_prompts[1] = format_message_to_role_mapper(system_prompts[1], 
                                                              role="system",
                                                              text=deepseek_reply)
            
            gpt_message = "This is what your rival had to say\n" + deepseek_reply

        else:
            system_prompts[1] = format_message_to_role_mapper(system_prompts[1],
                                                              role="user",
                                                              text=deepseek_message)

            deepseek_reply = deepseek(format_message_to_whole_strings(system_prompts[1]), 
                                       max_length=500,
                                       temperature=0.7)[0]["generated_text"]
            deepseek_reply = format_json_style(deepseek_reply, [])
            formatted_deepseek_reply = "This is what your rival had to say\n" + f"GPT : {deepseek_reply}"

            system_prompts[1] = format_message_to_role_mapper(system_prompts[1], 
                                                              role="system",
                                                              text=deepseek_reply)

            system_prompts[0] = format_message_to_role_mapper(system_prompts[0], 
                                                              role="user",
                                                              text=formatted_deepseek_reply)
            
            gpt_response = gpt.chat.completions.create(model=GPT_MODEL_NAME,
                                                       messages=system_prompts[0])
            gpt_reply = gpt_response.choices[0].message.content

            system_prompts[0] = format_message_to_role_mapper(system_prompts[0], 
                                                              role="system",
                                                              text=gpt_reply)
            
            deepseek_message = "This is what your rival had to say\n" + deepseek_reply

        rap_segments = judge_prompt + "Rap1: " + gpt_reply + "\nRap2: " + deepseek_reply
        system_prompts[2] = format_message_to_role_mapper(system_prompts[0],
                                                          role="user",
                                                          text=rap_segments)

        scores = judge.chat.completions.create(model=GPT_MODEL_NAME,
                                                messages=system_prompts[2])
        score_gpt += int(scores.split(",")[0])
        score_deepseek += int(scores.split(",")[1])

    winner = "GPT is the winner!" if score_gpt > score_deepseek else "Deepseek is the winner!"
    print(winner)


#### Begin circus ####
#pylint: disable=invalid-name
battle_rounds = 3

contestant_activation_content = "You are an intelligent assistant who can rap. \
                      Generate only the rap only in a json format\
                      where there will be a single key named 'system' \
                      and the value will be the rap itself."

judge_activation_content = "You are an intelligent assistant \
                            who can assign a score of 1 to 10 to a rap song."

gpt_system_prompt = [{"role" : "system", "content" : contestant_activation_content}]
deepseek_system_prompt = [{"role" : "system", "content" : contestant_activation_content}]
judge_system_prompt = [{"role" : "system", "content" : judge_activation_content}]
