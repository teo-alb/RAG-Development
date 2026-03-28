from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
def store_vec(embed, chunks):
    docs = [
        chunk if isinstance(chunk, Document) else Document(page_content=chunk)
        for chunk in chunks
    ]
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embed
    )
    
    print("Vectors stored:", vector_store.index.ntotal)

    # Save index locally
    vector_store.save_local(
        folder_path="faiss_index",
        index_name="CWC_index"
    )
    print("Index saved")

    # Load the stored index
    vector_store = FAISS.load_local(
        folder_path="faiss_index",
        index_name="CWC_index",
        embeddings=embed,
        allow_dangerous_deserialization=True
    )

    return vector_store