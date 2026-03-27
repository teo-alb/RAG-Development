import os

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from paths import model_patha



"""
========================
INDEXING PIPELINE
========================
"""

url = "https://example.com"

# Load HTML
loader = AsyncHtmlLoader(url)
data = loader.load()

# Convert HTML → text
transformer = Html2TextTransformer()
transformed_data = transformer.transform_documents(data)

print(transformed_data[0].page_content[:500])
print(transformed_data[0].metadata)


"""
Chunking
"""
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.create_documents(
    [transformed_data[0].page_content]
)

print("Number of chunks:", len(chunks))


"""
Embeddings
"""
embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


"""
Vector Store
"""
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embed
)

print("Vectors stored:", vector_store.index.ntotal)

vector_store.save_local(
    folder_path="faiss_index",
    index_name="CWC_index"
)

print("Index saved")


"""
========================
GENERATION PIPELINE
========================
"""

# Load vector store
vector_store = FAISS.load_local(
    folder_path="faiss_index",
    index_name="CWC_index",
    embeddings=embed,
    allow_dangerous_deserialization=True
)

# Query input
query = input("ENTER THE QUERY: ")

retrieved = vector_store.similarity_search(query, k=2)

print("\n--- Retrieved Context ---\n")
for doc in retrieved:
    print(doc.page_content)
    print("\n---\n")

# Build context
context = "\n\n".join([doc.page_content for doc in retrieved])

prompt = f"""
You are a strict QA system.

Rules:
- Answer ONLY using the provided context.
- Give EXACTLY ONE short answer.
- Do NOT repeat yourself.
- If the answer is not in the context, say: "I can't answer based on the provided context."

Question: {query}

Context:
{context}

Answer:
"""

"""
LLM SETUP (LlamaCpp)
"""
llm = LlamaCpp(
    model_path=model_patha,
    temperature=0.0,      # deterministic output → stops rambling
    max_tokens=20,        # only enough to answer the question
    n_ctx=1024,           # smaller context → faster
    n_threads=4,          # adjust to your CPU cores
    stop=["\n"]           # stops generation at the first newline
)

# Generate answer
response = llm.invoke(prompt)

print("\nAnswer:\n", response)