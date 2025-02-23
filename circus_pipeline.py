import os

import numpy as np

from utils.model_loading_utils import load_gpt, load_google_gemini
from utils.text_processing_utils import format_message_to_role_mapper
from utils.inference_utils import infer_openai_llms, decide_contest_result
from utils.constants import GEMINI_MODEL_NAME, GPT_MODEL_NAME


def begin_circus(system_prompts, rounds):
    """Main function to build AI circus and decide winner of the battle!"""

    gpt, system_prompts[0] = load_gpt(os.getenv("OPENAI_API_KEY"),
                                      GPT_MODEL_NAME, system_prompts[0])

    gemini, system_prompts[1] = load_google_gemini(os.getenv("OPENROUTER_KEY"),
                                                   GEMINI_MODEL_NAME, system_prompts[1])

    judge, system_prompts[2] = load_gpt(os.getenv("OPENAI_API_KEY"),
                                        GPT_MODEL_NAME, system_prompts[2])

    toss_and_pass = np.random.randint(2, size=1)[0]
    initial_prompt_part = "Let the rap battle begin! Whats your reply?"
    judge_prompt = "What would you rate these two raps out of 1 to 10? Return only the score\
                    delimited by a comma and nothing else.\n"

    gpt_message = gemini_message = initial_prompt_part
    score_gpt = score_gemini = 0

    for round_id in range(rounds):
        if toss_and_pass == 0:
            system_prompts[0] = format_message_to_role_mapper(system_prompts[0],
                                                              role="user",
                                                              text=gpt_message)

            gpt, system_prompts[0]  = infer_openai_llms(gpt, GPT_MODEL_NAME, system_prompts[0])
            gpt_reply = system_prompts[0][-1]["content"]
    
            formatted_gpt_reply = "This is what your rival had to say\n" + \
                                   f"GPT : {gpt_reply}"

            system_prompts[1] = format_message_to_role_mapper(system_prompts[1],
                                                              role="user",
                                                              text=formatted_gpt_reply)

            gemini, system_prompts[1] = infer_openai_llms(gemini, GEMINI_MODEL_NAME, system_prompts[1])
            gemini_reply = system_prompts[1][-1]["content"]
            gpt_message = "This is what your rival had to say\n" + gemini_reply

        else:
            system_prompts[1] = format_message_to_role_mapper(system_prompts[1],
                                                              role="user",
                                                              text=gemini_message)

            gemini, system_prompts[1] = infer_openai_llms(gemini, GEMINI_MODEL_NAME, system_prompts[1])
            gemini_reply = system_prompts[1][-1]["content"]

            formatted_gemini_reply = "This is what your rival had to say\n" +\
                                      f"Gemini : {gemini_reply}"

            system_prompts[0] = format_message_to_role_mapper(system_prompts[0], 
                                                              role="user",
                                                              text=formatted_gemini_reply)

            gpt, system_prompts[0]  = infer_openai_llms(gpt, GPT_MODEL_NAME, system_prompts[0])
            gpt_reply = system_prompts[0][-1]["content"]
            gemini_message = "This is what your rival had to say\n" + gpt_reply

        rap_segments = judge_prompt + "Rap1: " + gpt_reply + "\nRap2: " + gemini_reply

        print("ROUND : ", round_id + 1)
        print("Rap1 : ", gpt_reply)
        print()
        print("Rap2 : ", gemini_reply)
        print()

        system_prompts[2] = format_message_to_role_mapper(system_prompts[2],
                                                          role="user",
                                                          text=rap_segments)

        judge, system_prompts[2] = infer_openai_llms(judge, GPT_MODEL_NAME, 
                                                     system_prompts[2], True)
        scores = system_prompts[2][-1]["content"]
        score_gpt += int(scores.split(",")[0])
        score_gemini += int(scores.split(",")[1])

        print("Score GPT-4 : ", int(scores.split(",")[0]), "| "
              "Score GEMINI : ", int(scores.split(",")[1]))
        print()

    print("FINAL SCORE GEMINI : ", score_gemini)
    print("FINAL SCORE GPT-4 : ", score_gpt)
    decide_contest_result(score_gpt, score_gemini)


#### Begin circus ####
#pylint: disable=invalid-name
battle_rounds = 5

contestant_activation_content = "You are an intelligent assistant who can rap. \
                                Generate only the rap, in a json format\
                                where there will be a single key named 'system' \
                                and the value will be the rap itself. Reply with only yes or no \
                                if you understood your role."

judge_activation_content = "You are an intelligent assistant \
                            who can efficiently judge a rap and \
                            a score of 1 to 10 to a rap song. \
                            You have to be absolutely unbiased and \
                            assign the score to the raps based on \
                            the length of the rap, the rhythm of the rap, \
                            the roast level of the rap and the relevence of the rap."

gpt_system_prompt = [{"role" : "system", "content" : contestant_activation_content}]
deepseek_system_prompt = [{"role" : "system", "content" : contestant_activation_content}]
judge_system_prompt = [{"role" : "system", "content" : judge_activation_content}]

begin_circus(system_prompts=[gpt_system_prompt,
                             deepseek_system_prompt,
                             judge_system_prompt],
            rounds=battle_rounds)
