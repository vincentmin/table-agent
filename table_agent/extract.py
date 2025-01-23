from typing import Type
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from .graph import get_graph, Messages

type TableInput = pd.DataFrame

def load_default_model() -> BaseChatModel:
    """Loads a default model

    Returns:
        BaseChatModel: A default model
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model_name="gpt-4o-mini")



def extract[T](table: TableInput, output_model: Type[T], llm: BaseChatModel | None = None, prompt:str = "Extract data from table") -> tuple[str,list[T]]:
    """Takes in a table and an output model and returns a list of output models

    Args:
        table (TableInput): A pandas DataFrame
        output_model (Type[T]): A class that represents the output model

    Returns:
        tuple[str,list[T]]: A tuple of the response and the list of output models
    """
    if llm is None:
        llm = load_default_model()
    graph = get_graph(llm)
    messages: Messages = graph.invoke({
        "messages": [HumanMessage(text=prompt)],
        "df": table,
        "output_model": output_model
    })
    response = messages[-1]
    outputs: list[T] = messages[-2].artifact
    return response, outputs
