import sys
import os

import pandas as pd

from utils.model_loading_utils import load_gpt, load_google_gemini
from utils.text_processing_utils import format_message_to_role_mapper
from utils.inference_utils import infer_openai_llms
from utils.constants import RapPromptConst, PathConst


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def begin_circus(system_prompts, round_id, toss_winner, llm_replies):
    """Main function to process AI circus and decide the winner of the battle!"""

    candidates_df = pd.read_csv(PathConst.BATTLE_CANDIATES)
    first_contestant = candidates_df["first_contestant"].values[0]
    second_contestant = candidates_df["second_contestant"].values[0]
    competition_judge = candidates_df["judge"].values[0]

    gpt, system_prompts[0] = load_gpt(os.getenv("OPENAI_API_KEY"),
                                      first_contestant,
                                      system_prompts[0])

    gemini, system_prompts[1] = load_google_gemini(os.getenv("OPENROUTER_KEY"),
                                                   second_contestant,
                                                   system_prompts[1])

    judge, system_prompts[2] = load_gpt(os.getenv("OPENAI_API_KEY"),
                                        competition_judge,
                                        system_prompts[2])
    gpt_message = llm_replies[0]
    gemini_message = llm_replies[1]
    score_gpt = score_gemini = 0

    if toss_winner == 0:
        system_prompts[0] = format_message_to_role_mapper(system_prompts[0],
                                                            role="user",
                                                            text=gpt_message)
        gpt, system_prompts[0]  = infer_openai_llms(gpt,
                                                    first_contestant,
                                                    system_prompts[0])

        gpt_reply = "\n".join(system_prompts[0][-1]["content"].split("|"))
        formatted_gpt_reply = RapPromptConst.PROVOCATION_PROMPT + \
                              f"GPT : {gpt_reply}"

        system_prompts[1] = format_message_to_role_mapper(system_prompts[1],
                                                          role="user",
                                                          text=formatted_gpt_reply)
        gemini, system_prompts[1] = infer_openai_llms(gemini,
                                                      second_contestant,
                                                      system_prompts[1])

        gemini_reply = "\n".join(system_prompts[1][-1]["content"].split("|"))
        gpt_message = RapPromptConst.PROVOCATION_PROMPT + gemini_reply

    else:
        system_prompts[1] = format_message_to_role_mapper(system_prompts[1],
                                                          role="user",
                                                          text=gemini_message)
        gemini, system_prompts[1] = infer_openai_llms(gemini,
                                                      second_contestant,
                                                      system_prompts[1])

        gemini_reply = "\n".join(system_prompts[1][-1]["content"].split("|"))
        formatted_gemini_reply = RapPromptConst.PROVOCATION_PROMPT + \
                                 f"Gemini : {gemini_reply}"

        system_prompts[0] = format_message_to_role_mapper(system_prompts[0],
                                                          role="user",
                                                          text=formatted_gemini_reply)
        gpt, system_prompts[0]  = infer_openai_llms(gpt,
                                                    first_contestant,
                                                    system_prompts[0])

        gpt_reply = "\n".join(system_prompts[0][-1]["content"].split("|"))
        gemini_message = RapPromptConst.PROVOCATION_PROMPT + gpt_reply

    rap_segments = RapPromptConst.JUDGE_PROMPT + "Rap1: " + gpt_reply + "\nRap2: " + gemini_reply

    print("ROUND : ", round_id + 1)
    print("GPT REPLY : ", gpt_reply)
    print()
    print("GEMINI REPLY : ", gemini_reply)
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
