from dotenv import load_dotenv,find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(temperature=0.9)
prompt = ChatPromptTemplate.from_template(
"""
Translate the text from English to {language} .
text:'''
{text}
'''
"""
)
language="French"
text="I love you"
chain1 = LLMChain(llm=llm,prompt=prompt)
result1 = chain1.run({"language":language,"text":text})
print("result1 is :",result1)

