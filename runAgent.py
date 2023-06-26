import dateutil.utils
import langchain
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from langchain.agents import tool


@tool
def timeTool(text: str) -> str:
    """Returns toady's date,use this tool for any question
    about today's date.
    The input should always be an empty string,
    and the function will always return today's date
    """
    return str(dateutil.utils.today())


import langchain

langchain.debug = True
llm = OpenAI(temperature=0.9)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools + [timeTool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# query = "who won 2022 world cup"
# result = agent.run(query)
# print(result)

query = "who won 2022 world cup ? what date is today"
result = agent.run(query)
print(result)
