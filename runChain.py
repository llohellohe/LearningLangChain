from dotenv import load_dotenv,find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(temperature=0.9)
prompt = ChatPromptTemplate.from_template(
"""
Translate the text from English to French .
text:'''
{text}
'''
"""
)
language="French"
text="I love you"
chain1 = LLMChain(llm=llm,prompt=prompt)

prompt2 = ChatPromptTemplate.from_template(
"""
Translate the text from French to English .
text:'''
{text}
'''
"""
)

chain2 = LLMChain(llm=llm,prompt=prompt2)

from langchain.chains import SimpleSequentialChain
overall_simple_chain = SimpleSequentialChain(chains=[chain1,chain2],verbose=True)

overall_simple_chain.run(text)
