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
chain1 = LLMChain(llm=llm,prompt=prompt,output_key="translate_result")

prompt2 = ChatPromptTemplate.from_template(
"""
Translate the text from French to English .
text:'''
{translate_result}
'''
"""
)

chain2 = LLMChain(llm=llm,prompt=prompt2,output_key="tran_ret")

prompt3 = ChatPromptTemplate.from_template(
"""
Translate the text from English to Chinese .
text:'''
{tran_ret}
'''
"""
)

chain3 = LLMChain(llm=llm,prompt=prompt3,output_key="final_ret")


from langchain.chains import SequentialChain
overall_chain = SequentialChain(
    chains=[chain1,chain2,chain3],
    input_variables=["text","language"],
    output_variables=["translate_result","tran_ret","final_ret"],
    verbose=True)

result = overall_chain({"language":language,"text":text})
print(result)
print(result["final_ret"])