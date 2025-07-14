from dotenv import load_dotenv
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from typing import Union, List
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool

load_dotenv()


# create first tool
# IMPORTANT: description helps LLM decide whether to use this tool or not
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')  # strip away non-alphabetic characters
    return len(text)

def find_tool_by_names(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found...")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")

    tools = [get_text_length]

    # langsmith hub: hwchase17/react prompt
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:
    """

    # we only partially initialize the prompt with `tools` and `tool_names`
    # this prompt is not complete yet
    # because we haven't supplied it with our question (as a dict)
    prompt = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools), # cannot apply tool list directly, LLM only receives text inputs
                                                                     tool_names=", ".join([t.name for t in tools]))

    llm = ChatOpenAI(model="gpt-4.1-mini", # doesn't work on default model "gpt-3.5-turbo`
                     temperature=0.0,
                     stop=["\nObservation"], # stop generating text at this token
                     )

    agent = {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()
    # now populate the input placeholder
    # lambda function accessing the values of dict

    # invoke the chain
    # res = agent.invoke({"input": "what is the text length of 'Langchain Expression Language' in characters?"})
    # print(res)
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {"input": "what is the text length of 'Langchain Expression Language' in characters?"}
    )
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        # extrapolate the tool to use
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_names(tools, tool_name)
        # run the tool
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation}")


