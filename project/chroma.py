import os

from decouple import config
import chromadb
from google import genai
from groq import Groq
client=genai.Client(api_key=config("GEMINI_API_KEYS"))
groq_client=Groq(api_key=config("GROQ_API_KEY"))

def load_documents(path="./news_articles"):
    documents=[]
    for filename in os.listdir(path=path):
        with open(
            os.path.join(
                "./news_articles",filename),"r",encoding="utf8"
        ) as file:
            if filename.endswith('.txt'):
                documents.append({
                    "id":filename,
                    "text":file.read()
                })
    return documents

def get_embed_content(content):
    result=client.models.embed_content(
        model="gemini-embedding-2",
        contents=content
    )
    return result.embeddings[0].values

def split_text(text,chunk_size=1000,overlap=20):
    """helper function to split text into a list of character per item"""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def create_collection():
    chroma=chromadb.PersistentClient(path="./chroma_db")
    collection=chroma.get_or_create_collection(
        name="document_qa_collection",
    )
   
    return collection

def search_collection(collection:chromadb.Collection, query,n_result=2)->chromadb.QueryResult:
    query_embedding=get_embed_content(query)
    results=collection.query(
        query_embeddings=[query_embedding],
        n_results=n_result
    )
    return results
if __name__=="__main__":
    
    #load the document into memory
    documents=load_documents()
    #we split document into chunks
    
    chunks_document=[]
    for doc in documents:
        chunks=split_text(doc['text'])
        
        for i,chunk in enumerate(chunks):
            chunks_document.append({
                "id": f"{doc['id']}_chunk{i+1}", "text": chunk
            })
    
    #we then add the embedding to the actual chunks documents
    for chunks in chunks_document:
        chunks['embeddings']=get_embed_content(chunks['text'])
        
    #we create the collection
    collection=create_collection()
    
    #then add the 
    for doc in chunks_document:
        collection.upsert(
            ids=[doc['id']],
            embeddings=[doc['embeddings']],
            documents=[doc['text']]
        )    
    #at this point we can now receive the user query
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
            {"role": "system", "content": "You are a helpful assistant.Answer the question based on the context below. If the question can't be answered based on the context, say I don't know."},
            {
                "role": "user",
                "content": prompt
            } 
        
        ]
    )
    print("Generated response:", response.choices[0].message.content)
