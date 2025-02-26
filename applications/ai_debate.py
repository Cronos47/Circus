import sys
import os

import numpy as np
import pandas as pd

from utils.text_processing_utils import format_prompt
from utils.inference_utils import decide_contest_result
from utils.constants import ModelNameConst, DebatePromptConst
from utils.constants import CompetitionConst, PathConst

from competition_pipeline.circus_pipeline import begin_circus


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    #### Begin ai debate ####

    #pylint: disable=invalid-name
    gpt_system_prompt = [{"role" : "system",
                        "content" : format_prompt(DebatePromptConst.CONTESTANT_ACTIVATION_PROMPT)}]

    gemini_system_prompt = [{"role" : "system",
                            "content" : format_prompt(DebatePromptConst.CONTESTANT_ACTIVATION_PROMPT)}]

    judge_system_prompt = [{"role" : "system",
                            "content" : format_prompt(DebatePromptConst.JUDGE_ACTIVATION_PROMPT)}]

    input_prompts = [gpt_system_prompt,
                    gemini_system_prompt,
                    judge_system_prompt]

    first_contestant = ModelNameConst.GPT_MODEL_NAME
    second_contestant = ModelNameConst.GEMINI_PAID_MODEL_NAME
    competition_judge = ModelNameConst.UPGRADED_GPT_MODEL_NAME

    candidate_df = pd.DataFrame({"first_contestant" : [first_contestant],
                                 "second_contestant" : [second_contestant],
                                 "judge" : [competition_judge]})

    candidate_df.to_csv(PathConst.BATTLE_CANDIATES)

    gpt_output = gemini_output = DebatePromptConst.USER_PROMPT + \
                                 " " + DebatePromptConst.INITIAL_PROMPT_PART

    acc_score_gpt = acc_score_gemini = 0
    toss_and_pass = np.random.randint(2, size=1)[0]

    if toss_and_pass == 0:
        gpt_output = gpt_output + " " + DebatePromptConst.POSITIVE_SIDE_PROMPT
        gemini_output = gemini_output + " " + DebatePromptConst.NEGATIVE_SIDE_PROMPT

    else:
        gpt_output = gpt_output + " " + DebatePromptConst.NEGATIVE_SIDE_PROMPT
        gemini_output = gemini_output + " " + DebatePromptConst.POSITIVE_SIDE_PROMPT

    for round_idx in range(CompetitionConst.BATTLE_ROUNDS):
        response, input_prompts = begin_circus(system_prompts=input_prompts,
                                               round_id=round_idx,
                                               toss_winner=toss_and_pass,
                                               llm_replies=[gpt_output, gemini_output])
        acc_score_gpt += response["gpt_score"]
        acc_score_gemini += response["gemini_score"]
        gpt_output = response["gpt_reply"]
        gemini_output = response["gemini_reply"]

    os.remove(PathConst.BATTLE_CANDIATES)

    print("FINAL SCORE OF GPT-4: ", acc_score_gpt)
    print("FINAL SCORE OF GEMINI: ", acc_score_gemini)
    decide_contest_result(acc_score_gpt, acc_score_gemini)
