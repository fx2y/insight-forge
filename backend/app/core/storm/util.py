from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores import SKLearnVectorStore

from app.core.config import JINA_API_KEY

wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, top_k_results=1)
embeddings = JinaEmbeddings(jina_api_key=JINA_API_KEY, model="jina-embeddings-v3")
vectorstore = SKLearnVectorStore(embedding=embeddings)
retriever = vectorstore.as_retriever(k=10)
