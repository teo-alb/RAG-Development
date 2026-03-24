from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

"""
Load HTML from url and transform it into plain text
"""

url = "url" #here goes the link
loader = AsyncHtmlLoader(url)
data = loader.load()

#print(data)

transformer = Html2TextTransformer()
transformed_data = transformer.transform_documents(data)
print(transformed_data[0].page_content)     
print(transformed_data[0].metadata)         