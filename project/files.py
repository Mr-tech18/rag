import os
def split_text(text,chunk_size=1000,overlap=20):
    """helper function to split text into a list of character per item"""
    chunk=[]
    start=0
    while start <chunk_size:
        end=start+chunk_size
        chunk.append(text[start:end])
        start=end-overlap
    return chunk

if __name__=="__main__":
    document=[]
    for filename in os.listdir(path="./news_articles"):
        with open(
            os.path.join(
                "./news_articles",filename),"r",encoding="utf8"
        ) as file:
            if filename.endswith('.txt'):
                document.append({
                    "id":filename,
                    "content":file.read()
                })
    chunked_documents=[]
    
    for doc in document:
        chunks=split_text(doc['content'])
        
        for i,chunk in enumerate(chunks):
            chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

    print(chunked_documents[1])