def augment(retrieved,query):
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
    """

    return prompt