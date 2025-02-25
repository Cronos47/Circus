class ModelNameConst:
    """Class structure of all the model name constants"""

    DEEPSEEK_MODEL_NAME = "deepseek-ai/deepseek-llm-67b-chat"
    DEEPSEEK_OPENAI_NAME = "deepseek/deepseek-r1-distill-llama-70b:free"
    LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    GEMINI_MODEL_NAME = "google/gemini-2.0-flash-lite-preview-02-05:free"
    GEMINI_PAID_MODEL_NAME = "google/gemini-2.0-flash-001"
    MISTRAL_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    GPT_MODEL_NAME = "gpt-4o-mini"
    UPGRADED_GPT_MODEL_NAME = "gpt-4o"


class PromptConst:
    """Class structure of all the prompt related constants"""

    INITIAL_PROMPT_PART = "Let the rap battle begin! Whats your reply?"

    JUDGE_PROMPT = "What would you rate these two raps out of 1 to 10? Return only the scores\
                    delimited by a comma and nothing else.\n"

    CONTESTANT_ACTIVATION_PROMPT = 'You are an intelligent assistant who can rap. \
                                    Generate only a json containing the rap and nothing else.\
                                    The json will have a single key named "system" and the value will be the rap itself \
                                    so the structure will be as follows {"system": "<the rap lines>"}. \
                                    Always make sure to keep the entire rap within double quotes \
                                    and please do not use any double quotes (" ") inside any of the rap lines \
                                    if you have to highlight something then use only single quotes (' ') for that.\
                                    and for line breaks use the "|" character,\
                                    also do not use any escape characters, no backslash character neither any newline character and \
                                    please make sure not to repeat the same rap ever. \
                                    And before returning the json make sure that the json is correctly formatted in such a way \
                                    that it can be read by pythons json.loads() method without errors. \
                                    Also keep in mind that you cannot copy off of your opponent.\
                                    Reply with only yes or no if you understood your role.'

    JUDGE_ACTIVATION_PROMPT = "You are a rap battle judge. You will score two rap songs out of 1 to 10\
                               based on the following criteria: \
                               1. **Length** – Consider the number of lines and overall detail. Longer and more detailed raps score higher. \
                               2. **Roasting Level** – Evaluate the creativity and intensity of the disses. The more clever and brutal, the higher the score.\
                               3. **Meaningful Depth** – Consider how insightful, clever, or thought-provoking the lyrics are. \
                               Each rap will receive a score from **1 to 10** for each criterion. \
                               Calculate an **average score** for each rap based on the three factors.\
                                **Rules for Scoring:** \
                                - **Be objective** based on the given criteria.\
                                - **Return ONLY the final scores** as **`score1,score2`**.  \
                                - **No extra commentary, explanations, or text.** \
                                ### Example:\
                                **Rap 1:**  \
                                'I'm the king of the throne, never leaving my zone,  \
                                Your bars are weak, man, I shatter your tone.'\
                                **Rap 2:**  \
                                'Your flow's outdated, stuck in the past,  \
                                I'm futuristic, spitting rhymes that last.' \
                                **Output:**  \
                                8,7"

    PROVOCATION_PROMPT = "This is what your rival had to say\n"


class CompetitionConst:
    """Class structure for globally relevant compeition constants"""

    BATTLE_ROUNDS = 15
