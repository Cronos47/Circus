import os

import numpy as np

from utils.model_loading_utils import load_gpt, load_google_gemini
from utils.text_processing_utils import format_message_to_role_mapper
from utils.inference_utils import infer_openai_llms, decide_contest_result
from utils.constants import ModelNameConst, PromptConst, CompetitionConst


def begin_circus(system_prompts, rounds):
    """Main function to process AI circus and decide the winner of the battle!"""

    gpt, system_prompts[0] = load_gpt(os.getenv("OPENAI_API_KEY"),
                                      ModelNameConst.GPT_MODEL_NAME,
                                      system_prompts[0])

    gemini, system_prompts[1] = load_google_gemini(os.getenv("OPENROUTER_KEY"),
                                                   ModelNameConst.MISTRAL_MODEL_NAME,
                                                   system_prompts[1])

    judge, system_prompts[2] = load_gpt(os.getenv("OPENAI_API_KEY"),
                                        ModelNameConst.GPT_MODEL_NAME,
                                        system_prompts[2])

    toss_and_pass = np.random.randint(2, size=1)[0]
    gpt_message = gemini_message = PromptConst.INITIAL_PROMPT_PART
    score_gpt = score_gemini = 0

    for round_id in range(rounds):
        if toss_and_pass == 0:
            system_prompts[0] = format_message_to_role_mapper(system_prompts[0],
                                                              role="user",
                                                              text=gpt_message)

            gpt, system_prompts[0]  = infer_openai_llms(gpt,
                                                        ModelNameConst.GPT_MODEL_NAME,
                                                        system_prompts[0])
            gpt_reply = system_prompts[0][-1]["content"]

            formatted_gpt_reply = PromptConst.PROVOCATION_PROMPT + \
                                  f"GPT : {gpt_reply}"

            system_prompts[1] = format_message_to_role_mapper(system_prompts[1],
                                                              role="user",
                                                              text=formatted_gpt_reply)

            gemini, system_prompts[1] = infer_openai_llms(gemini,
                                                          ModelNameConst.MISTRAL_MODEL_NAME,
                                                          system_prompts[1])

            gemini_reply = system_prompts[1][-1]["content"]
            gpt_message = PromptConst.PROVOCATION_PROMPT + gemini_reply

        else:
            system_prompts[1] = format_message_to_role_mapper(system_prompts[1],
                                                              role="user",
                                                              text=gemini_message)

            gemini, system_prompts[1] = infer_openai_llms(gemini,
                                                          ModelNameConst.MISTRAL_MODEL_NAME,
                                                          system_prompts[1])
            gemini_reply = system_prompts[1][-1]["content"]

            formatted_gemini_reply = PromptConst.PROVOCATION_PROMPT +\
                                     f"Gemini : {gemini_reply}"

            system_prompts[0] = format_message_to_role_mapper(system_prompts[0],
                                                              role="user",
                                                              text=formatted_gemini_reply)
            gpt, system_prompts[0]  = infer_openai_llms(gpt,
                                                        ModelNameConst.GPT_MODEL_NAME,
                                                        system_prompts[0])
            gpt_reply = system_prompts[0][-1]["content"]
            gemini_message = PromptConst.PROVOCATION_PROMPT + gpt_reply

        rap_segments = PromptConst.JUDGE_PROMPT + "Rap1: " + gpt_reply + "\nRap2: " + gemini_reply

        print("ROUND : ", round_id + 1)
        print("GPT RAP : ", gpt_reply)
        print()
        print("GEMINI RAP : ", gemini_reply)
        print()

        system_prompts[2] = format_message_to_role_mapper(system_prompts[2],
                                                          role="user",
                                                          text=rap_segments)
        judge, system_prompts[2] = infer_openai_llms(judge,
                                                     ModelNameConst.GPT_MODEL_NAME,
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
gpt_system_prompt = [{"role" : "system", "content" : PromptConst.CONTESTANT_ACTIVATION_PROMPT}]
gemini_system_prompt = [{"role" : "system", "content" : PromptConst.CONTESTANT_ACTIVATION_PROMPT}]
judge_system_prompt = [{"role" : "system", "content" : PromptConst.JUDGE_ACTIVATION_PROMPT}]

begin_circus(system_prompts=[gpt_system_prompt,
                             gemini_system_prompt,
                             judge_system_prompt],
            rounds=CompetitionConst.BATTLE_ROUNDS)
