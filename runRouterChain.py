from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

_ = load_dotenv(find_dotenv())

physics_template = """You are a very smart physics professor.
You are great at answering question about physics.
When you don't know the answer to a question you admit that.
Here is a question:
{input}
"""

math_template = """You are a very smart mathematician .
You are great at answering question about math.
When you don't know the answer to a question you admit that.
Here is a question:
{input}
"""

history_template = """You are a very good historian.
You are great at answering question about history.
When you don't know the answer to a question you admit that.
Here is a question:
{input}
"""

computer_science_template = """You are a very smart computer scientist.
You are great at answering question about computer science.
When you don't know the answer to a question you admit that.
Here is a question:
{input}
"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering questions about math",
        "prompt_template": math_template
    },
    {
        "name": "history",
        "description": "Good for answering questions about history",
        "prompt_template": history_template
    },
    {
        "name": "computer science",
        "description": "Good for answering questions about computer science",
        "prompt_template": computer_science_template
    },
]

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0.9)
destination_chains = {}

for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}:{p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

print(destinations)
print(destinations_str)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """
Given a row text input ,you need to try to finger out
<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted
```json
{{{{
    "destination": string \ name of the prompt to choose
    "next_inputs": string \ a potentially input text for the prompt
}}}}
```
REMEMBER: "destination" MUST be one of the candidate prompts
REMEMBER: "next_inputs"  can just be the original input for the prompts
<< CANDIDATE PROMPTS >>
{destinations}
<< INPUT >>
{{input}}
<< OUTPUT (remember to include the ```json) >>
"""
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
print(router_template)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain,
                         verbose=True)

##chain.run("what is 1+1 ?")
result = chain.run("first world war start from which year ?")
print(result)
