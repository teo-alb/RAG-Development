from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking(text:tuple):
    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=500,
    chunk_overlap=100
    )

    chunks = text_splitter.create_documents(
    [text[0].page_content]
    )

    return chunks