import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from decouple import config

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", api_key=config("GEMINI_API_KEY"))



def test():
    vectorstore = chromadb.Client().get_or_create_collection("my_collection")
    results = vectorstore.query(query_embeddings=embeddings.embed_query("What is LangChain? and what is langgraph?"), n_results=3)    

    
    print(f"Content: {results['documents']}, Metadata: {results['metadatas']}")
    print("-----")

if __name__ == "__main__":
    test()
    