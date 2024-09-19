from langchain_openai import ChatOpenAI

from app.core.config import MODEL_NAME, OPENAI_API_BASE, OPENAI_API_KEY

fast_llm = ChatOpenAI(model=MODEL_NAME, openai_api_base=OPENAI_API_BASE, openai_api_key=OPENAI_API_KEY)
long_context_llm = ChatOpenAI(model=MODEL_NAME, openai_api_base=OPENAI_API_BASE, openai_api_key=OPENAI_API_KEY)
