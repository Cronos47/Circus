import os

import numpy as np

from utils.model_loading_utils import load_gpt, load_google_gemini
from utils.text_processing_utils import format_message_to_role_mapper, format_prompt
from utils.inference_utils import infer_openai_llms, decide_contest_result
from utils.constants import ModelNameConst, PromptConst, CompetitionConst


def begin_circus(system_prompts, round_id, toss_winner):
    """Main function to process AI circus and decide the winner of the battle!"""

    gpt, system_prompts[0] = load_gpt(os.getenv("OPENAI_API_KEY"),
                                      first_contestant,
                                      system_prompts[0])

    gemini, system_prompts[1] = load_google_gemini(os.getenv("OPENROUTER_KEY"),
                                                   second_contestant,
                                                   system_prompts[1])

    judge, system_prompts[2] = load_gpt(os.getenv("OPENAI_API_KEY"),
                                        competition_judge,
                                        system_prompts[2])

    gpt_message = gemini_message = PromptConst.INITIAL_PROMPT_PART
    score_gpt = score_gemini = 0
    
    if toss_winner == 0:
        system_prompts[0] = format_message_to_role_mapper(system_prompts[0],
                                                            role="user",
                                                            text=gpt_message)
        gpt, system_prompts[0]  = infer_openai_llms(gpt,
                                                    first_contestant,
                                                    system_prompts[0])

        gpt_reply = "\n".join(system_prompts[0][-1]["content"].split("|"))
        formatted_gpt_reply = PromptConst.PROVOCATION_PROMPT + \
                              f"GPT : {gpt_reply}"

        system_prompts[1] = format_message_to_role_mapper(system_prompts[1],
                                                          role="user",
                                                          text=formatted_gpt_reply)
        gemini, system_prompts[1] = infer_openai_llms(gemini,
                                                      second_contestant,
                                                      system_prompts[1])

        gemini_reply = "\n".join(system_prompts[1][-1]["content"].split("|"))
        gpt_message = PromptConst.PROVOCATION_PROMPT + gemini_reply

    else:
        system_prompts[1] = format_message_to_role_mapper(system_prompts[1],
                                                          role="user",
                                                          text=gemini_message)
        gemini, system_prompts[1] = infer_openai_llms(gemini,
                                                      second_contestant,
                                                      system_prompts[1])

        gemini_reply = "\n".join(system_prompts[1][-1]["content"].split("|"))
        formatted_gemini_reply = PromptConst.PROVOCATION_PROMPT + \
                                 f"Gemini : {gemini_reply}"

        system_prompts[0] = format_message_to_role_mapper(system_prompts[0],
                                                          role="user",
                                                          text=formatted_gemini_reply)
        gpt, system_prompts[0]  = infer_openai_llms(gpt,
                                                    first_contestant,
                                                    system_prompts[0])

        gpt_reply = "\n".join(system_prompts[0][-1]["content"].split("|"))
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
                                                 competition_judge,
                                                 system_prompts[2], True)
    scores = system_prompts[2][-1]["content"]
    score_gpt += int(scores.split(",")[0])
    score_gemini += int(scores.split(",")[1])

    print("Score GPT-4 : ", int(scores.split(",")[0]), "| "
            "Score GEMINI : ", int(scores.split(",")[1]))
    print()
    round_response = {
                        "round" : round_id + 1,
                        "gpt_reply" : gpt_reply,
                        "gemini_reply" : gemini_reply,
                        "gpt_score" : int(scores.split(",")[0]),
                        "gemini_score" : int(scores.split(",")[1])
                     }
    return round_response, system_prompts.copy()


#### Begin circus ####

#pylint: disable=invalid-name
gpt_system_prompt = [{"role" : "system",
                      "content" : format_prompt(PromptConst.CONTESTANT_ACTIVATION_PROMPT)}]

gemini_system_prompt = [{"role" : "system",
                         "content" : format_prompt(PromptConst.CONTESTANT_ACTIVATION_PROMPT)}]

judge_system_prompt = [{"role" : "system",
                        "content" : format_prompt(PromptConst.JUDGE_ACTIVATION_PROMPT)}]

input_prompts = [gpt_system_prompt,
                gemini_system_prompt,
                judge_system_prompt]

first_contestant = ModelNameConst.GPT_MODEL_NAME
second_contestant = ModelNameConst.GEMINI_PAID_MODEL_NAME
competition_judge = ModelNameConst.UPGRADED_GPT_MODEL_NAME

acc_score_gpt = acc_score_gemini = 0
toss_and_pass = np.random.randint(2, size=1)[0]

for round_idx in range(CompetitionConst.BATTLE_ROUNDS):
    response, input_prompts = begin_circus(system_prompts=input_prompts,
                                            round_id=round_idx,
                                            toss_winner=toss_and_pass)
    acc_score_gpt += response["gpt_score"]
    acc_score_gemini += response["gemini_score"]

print("FINAL SCORE OF GPT-4: ", acc_score_gpt)
print("FINAL SCORE OF GEMINI: ", acc_score_gemini)
decide_contest_result(acc_score_gpt, acc_score_gemini)
