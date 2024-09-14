import langroid.language_models as lm
from langroid.utils.configuration import settings

CONTEXT_LENGTH = 128000

settings.debug = False
settings.cache = False

"""
LLM_CONFIGS: dict
    A dictionary of OpenAIGPTConfig objects for different LLM models.
    The keys are the model names and the values are the OpenAIGPTConfig objects.
"""
LLM_CONFIGS = {
    "medium": lm.OpenAIGPTConfig(
        chat_model="ollama/gemma2:27b",
        chat_context_length=CONTEXT_LENGTH,
        max_output_tokens=1000,
        temperature=0.2,
        stream=True,
        timeout=180,
    ),
    "small": lm.OpenAIGPTConfig(
        chat_model="ollama/llama3.1:8b",
        chat_context_length=CONTEXT_LENGTH,
        max_output_tokens=1000,
        temperature=0.2,
        stream=True,
        timeout=180,
    ),
    "tiny": lm.OpenAIGPTConfig(
        chat_model="ollama/phi3:mini",
        chat_context_length=CONTEXT_LENGTH,
        max_output_tokens=1000,
        temperature=0.2,
        stream=True,
        timeout=180,
    ),
}
