from pypdf import PdfReader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

def extract_text(path="data/microsoft-annual-report.pdf"):


    reader= PdfReader(path)

    print("Number of pages in the PDF:", len(reader.pages))

    print("Extracting text from each page:")
    print(reader.metadata)
    print(reader.is_encrypted)
    pdf_texts = [p.extract_text() for p in reader.pages]

    #pdf_texts=[text for text in pdf_texts if text]
    print(pdf_texts[0])
    print('pdf text------>')
    return pdf_texts

def text_splitter(text:list):
    text_s=RecursiveCharacterTextSplitter(
        chunk_size=1000,
    )
    
    return text_s.split_text('\n'.join(text))



if __name__=="__main__":
    

    
    text_list=text_splitter(extract_text())
    character_text_splitter=SentenceTransformersTokenTextSplitter(tokens_per_chunk=256)
    chunk_splitters=[]
    for chunk in text_list:
        chunk=character_text_splitter.split_text(chunk)
        chunk_splitters.append(chunk)
        

    # now we import chromadb and the SentenceTransformerEmbeddingFunction
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


    embedding_function = SentenceTransformerEmbeddingFunction()
    chunk_splitters=[chunk for sublist in chunk_splitters for chunk in sublist]
    print(embedding_function([chunk_splitters[10]]))

 


