from dotenv import load_dotenv
from langchain.agents import tool


load_dotenv()

# create first tool
# IMPORTANT: description helps LLM decide whether to use this tool or not
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"') # strip away non-alphabetic characters
    return len(text)




if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    print(get_text_length.invoke(input={"text": "LangChain Expression Language"}))
