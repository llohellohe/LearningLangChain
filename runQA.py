from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())


from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator


loader = TextLoader('state_of_the_union.txt', encoding="utf-8")

index = VectorstoreIndexCreator().from_loaders([loader])
query = "What did the president say about Ketanji Brown Jackson"
print(index.query(query))





