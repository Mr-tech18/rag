from decouple import config

import chromadb
from google import genai
from groq import Groq

from chroma import get_embed_content, search_collection

collection=chromadb.PersistentClient(path="./chroma_db").get_collection(
    name="document_qa_collection",
)

client=genai.Client(api_key=config("GEMINI_API_KEYS"))
groq_client=Groq(api_key=config("GROQ_API_KEY"))

def query_documents(question, n_results=2):
    # compute embedding using the same embedding function used when upserting
    query_embedding = get_embed_content(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    print(results['documents'])
    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks

question="tell me about databricks"
context = "\n\n".join(query_documents(question))

results=search_collection(collection, question)
prompt = (
    "You are an assistant for question-answering tasks. Use the following pieces of "
    "retrieved context to answer the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the answer concise."
    "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
)
response=groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": question,
        } 
    
    ]
)
print("Generated response:", response.choices[0].message.content)
