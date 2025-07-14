# 🦜🔗 ReAct Agent **from Scratch** with LangChain

> A minimal, fully-transparent implementation of a ReAct `AgentExecutor` built step-by-step in plain Python + LangChain.

<!-- Badges -->
![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-9cf?logo=langchain)
![license](https://img.shields.io/badge/license-MIT-lightgrey)
[![LangSmith](https://img.shields.io/badge/LangSmith-enabled-brightgreen)](https://smith.langchain.com)

---

---

## 📈 LangSmith Tracing

LangSmith tracing: [Link](https://smith.langchain.com/o/856312b1-7816-4389-80cb-b01e398655be/projects/p/853ae249-a061-4fe7-b294-ff0dce1eb14e?timeModel=%7B%22duration%22%3A%227d%22%7D)

## 💡 Quick Demo

```bash
# 1. clone & move inside
git clone https://github.com/ndkhoa211/ReAct-agent-from-scratch.git
cd ReAct-agent-from-scratch

# 2. create an isolated env (recommended)
uv venv          # creates .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. install runtime deps
uv pip install -e .

# 4. add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# 5. run it!
python main.py
```

Output:
```
Hello ReAct LangChain!
***Prompt to LLM was:***
Human: 
    Answer the following questions as best you can. You have access to the following tools:

    get_text_length(text: str) -> int - Returns the length of a text by characters

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [get_text_length]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: what is the text length of 'Langchain Expression Language' in characters?
    Thought: 
    
**********
***LLM Response:***
Question: what is the text length of 'Langchain Expression Language' in characters?
Thought: I need to find the number of characters in the text 'Langchain Expression Language'.
Action: get_text_length
Action Input: Langchain Expression Language
**********
agent_step=tool='get_text_length' tool_input='Langchain Expression Language' log="Question: what is the text length of 'Langchain Expression Language' in characters?\nThought: I need to find the number of characters in the text 'Langchain Expression Language'.\nAction: get_text_length\nAction Input: Langchain Expression Language"
*****
*****agent_step is an instance of AgentAction*****
*****
get_text_length enter with text='Langchain Expression Language'
29
***Prompt to LLM was:***
Human: 
    Answer the following questions as best you can. You have access to the following tools:

    get_text_length(text: str) -> int - Returns the length of a text by characters

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [get_text_length]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: what is the text length of 'Langchain Expression Language' in characters?
    Thought: Question: what is the text length of 'Langchain Expression Language' in characters?
Thought: I need to find the number of characters in the text 'Langchain Expression Language'.
Action: get_text_length
Action Input: Langchain Expression Language
Observation: 29
Thought: 
    
**********
***LLM Response:***
Thought: I now know the final answer
Final Answer: The text length of 'Langchain Expression Language' is 29 characters.
**********
agent_step=return_values={'output': "The text length of 'Langchain Expression Language' is 29 characters."} log="Thought: I now know the final answer\nFinal Answer: The text length of 'Langchain Expression Language' is 29 characters."
*****
*****agent_step is an instance of AgentFinish*****
*****
{'output': "The text length of 'Langchain Expression Language' is 29 characters."}
```
