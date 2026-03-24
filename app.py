from langchain_community.document_loaders import AsyncHtmlLoader
url = "https://packaging.python.org/en/latest/"
loader = AsyncHtmlLoader(url)
data = loader.load()
print(data)