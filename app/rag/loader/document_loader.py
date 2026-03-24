from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.heum.ai/ko/apply-faq")

docs = loader.load()

print(docs)