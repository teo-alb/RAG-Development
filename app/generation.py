from langchain_community.llms import LlamaCpp
import the_constants

def generate(prompt):
    llm = LlamaCpp(
        model_path=the_constants.MODEL_PATH,
        temperature=the_constants.TEMPERATURE,
        max_tokens=the_constants.TOKENS,
        n_ctx=the_constants.CONTEXT_TOKENS,
        n_threads=the_constants.THREADS,
        stop=the_constants.STOP_S
    )

    response = llm.invoke(prompt)
    print("\nAnswer:\n", response)