import os

from dotenv import load_dotenv

load_dotenv("../../../.env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

JINA_API_KEY = os.getenv("JINA_API_KEY")