from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer


def loader(url):

    loader = AsyncHtmlLoader(url)
    raw_data = loader.load()

    transformer = Html2TextTransformer()
    data = transformer.transform_documents(raw_data)
    #print(data)
    return data