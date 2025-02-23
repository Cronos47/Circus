class ModelNameConst:
    """Class structure of all the model name constants"""

    DEEPSEEK_MODEL_NAME = "deepseek-ai/deepseek-llm-67b-chat"
    DEEPSEEK_OPENAI_NAME = "deepseek/deepseek-r1-distill-llama-70b:free"
    LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    GEMINI_MODEL_NAME = "google/gemini-2.0-flash-lite-preview-02-05:free"
    MISTRAL_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    GPT_MODEL_NAME = "gpt-4o-mini"


class PromptConst:
    """Class structure of all the prompt related constants"""

    INITIAL_PROMPT_PART = "Let the rap battle begin! Whats your reply?"

    JUDGE_PROMPT = "What would you rate these two raps out of 1 to 10? Return only the scores\
                    delimited by a comma and nothing else.\n"

    CONTESTANT_ACTIVATION_PROMPT = "You are an intelligent assistant who can rap. \
                                    Generate only the rap, in a json format\
                                    where there will be a single key named 'system' \
                                    and the value will be the rap itself and \
                                    before returning the json make sure \
                                    the json is well-formatted and \
                                    can be read in python without error. \
                                    Reply with only yes or no if you understood your role."

    JUDGE_ACTIVATION_PROMPT = "You are an intelligent assistant \
                                who can efficiently judge a rap and \
                                a score of 1 to 10 to a rap song. \
                                You have to be absolutely unbiased and \
                                assign the score to the raps based on \
                                the length of the rap, the rhythm of the rap, \
                                the roast level of the rap and the relevence of the rap."
    
    PROVOCATION_PROMPT = "This is what your rival had to say\n"


class CompetitionConst:
    """Class structure for globally relevant compeition constants"""

    BATTLE_ROUNDS = 10
