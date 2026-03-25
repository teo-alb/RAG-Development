from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
"""
Load HTML from url and transform it into plain text
"""

url = "https://example.com" #here goes the url
loader = AsyncHtmlLoader(url)
data = loader.load()

#print(data)

transformer = Html2TextTransformer()
transformed_data = transformer.transform_documents(data)
print(transformed_data[0].page_content)     
print(transformed_data[0].metadata)         

"""
Chunking - Splitting the text
"""

text_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", ".", " "],
    chunk_size = 1000,
    chunk_overlap = 200    #20% overlap
)

chunks = text_splitter.create_documents(
    [transformed_data[0].page_content]
)

print("Number of chunks is ", len(chunks)) #number of chunks created


embed = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2"
)
embeddings = embed.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print(embeddings)


"""
Create the vector database
"""

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embed
)
print("Vectors stored:", vector_store.index.ntotal)