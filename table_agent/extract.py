from typing import Type, TypedDict, TypeVar
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from .graph import get_graph
from .types import State

T = TypeVar("T")

type TableInput = pd.DataFrame


class TableOutput(TypedDict):
    response: str
    script: str
    outputs: list[T]


def load_default_model() -> BaseChatModel:
    """Loads a default model

    Returns:
        BaseChatModel: A default model
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model_name="gpt-4o-mini")


def extract[T](
    table: TableInput,
    output_model: Type[T],
    llm: BaseChatModel | None = None,
    prompt: str = "Extract data from table",
) -> TableOutput:
    """Takes in a table and an output model and returns a list of output models

    Args:
        table (TableInput): A pandas DataFrame
        output_model (Type[T]): A class that represents the output model

    Returns:
        tuple[str,list[T]]: A tuple of the response and the list of output models
    """
    if llm is None:
        llm = load_default_model()
    graph = get_graph(llm, table, output_model)
    state: State = graph.invoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "df": table,
            "output_model": output_model,
        }
    )
    response = state["messages"][-1]
    artifact: tuple[str, list[T]] | None = state["messages"][-2].artifact
    if artifact is None:
        script = ""
        outputs = []
    else:
        script, outputs = artifact
    return {
        "response": response.content,
        "script": script,
        "outputs": outputs,
    }
