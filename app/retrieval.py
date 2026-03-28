def retrieval(vector_store):
    query = input("ENTER THE QUERY: ")
    retrieved = vector_store.similarity_search(query, k=2)

    print("\n--- Retrieved Context ---\n")
    for doc in retrieved:
        print(doc.page_content)
        print("\n---\n")

    return retrieved,query