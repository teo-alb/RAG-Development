from langchain_huggingface import HuggingFaceEmbeddings

def embeddings(chunk):
    embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return embed
