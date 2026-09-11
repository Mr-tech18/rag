from json import load
from os import pipe

from langchain_docling.loader import DoclingLoader
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, CodeFormulaVlmOptions
from sklearn import pipeline
from dotenv import load_dotenv
from mineru import Min
load_dotenv()

pipeline_options = PdfPipelineOptions()
pipeline_options.do_formula_enrichment = True

converter=DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def doc_loader():
    loader=DoclingLoader(file_path="./data/math_test.pdf",
                         converter=converter
                        )
    
    docs=loader.load()
    
    print(f"Number of documents loaded: {len(docs)}")
    print(f"Document content: {docs[1].page_content[:500]}")  # Print first 500 characters of the first document
    
    
    

if __name__ == "__main__":
    doc_loader()