import chromadb

chroma_client=chromadb.Client()

collection = chroma_client.get_or_create_collection("my_collection")

documents=[
    {"id": "1", "text": "This is the first document."},
    {"id": "2", "text": "This is the second document."},
    {"id": "3", "text": "This is the third document."}]
query=" first document."

for doc in documents:
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]]
    )

response=collection.query(
    query_texts=[query],
    n_results=2)

print(response.get("results", []))