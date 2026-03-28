from loading import loader
from chunking import chunking
from embeddings import embeddings
from vector_store import store_vec
from retrieval import retrieval
from augmentation import augment
from generation import generate

def main():
    print("Hello RAG - The system is running")
    data = loader("https://example.com")  # tuple
    print(data)

    chunks = chunking(data)  # list
    print("THE CHUNK CONTAINS :-----", chunks)

    embed_vectors = embeddings(chunks)

    vector_store = store_vec(embed_vectors, chunks)

    retrieved,query = retrieval(vector_store)

    final_prompt = augment(retrieved,query)

    generate(final_prompt)

    
if __name__ == "__main__":
    main()