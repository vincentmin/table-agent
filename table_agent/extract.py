from typing import Type, TypedDict
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from .graph import get_graph
from .types import State


class TableOutput[T](TypedDict):
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


SYSTEM_PROMPT = """Your job is to extract data from a table by writing a python script.
The python script must read the table as a pandas dataframe from `table.parquet`.
Then write code that parses each row of the dataframe into a json with the following json schema format:
{schema}

The script should write the parsed jsons to a file called `output.json`.
Make sure that `ouput.json` is a list of jsons that comply with the above format.

Here follow the first 5 lines of the dataframe you need to act on:
{df}"""


def extract[T: BaseModel](
    table: pd.DataFrame,
    output_model: Type[T],
    llm: BaseChatModel | None = None,
    system_prompt: str | None = None,
    user_prompt: str = "Extract data from table",
    config: RunnableConfig | None = None,
) -> TableOutput[T]:
    """Takes in a table and an output model and returns a list of output models

    Args:
        table (pd.DataFrame): A pandas DataFrame
        output_model (Type[T]): A pydantic model that the table will be parsed into
        llm (BaseChatModel, optional): A language model. Defaults to None.
        system_prompt (str, optional): The system prompt. Defaults to None.
        user_prompt (str, optional): The user prompt. Defaults to "Extract data from table".
        config (RunnableConfig, optional): The configuration. Defaults to None.

    Returns:
        TableOutput: The model response, the script, and the outputs
    """

    llm = llm or load_default_model()
    system_prompt = system_prompt or SYSTEM_PROMPT.format(
        df=table.head(), schema=output_model.model_json_schema()
    )
    graph = get_graph(llm, table, output_model)
    state: State = graph.invoke(
        {
            "messages": [SystemMessage(system_prompt), HumanMessage(user_prompt)],
            "df": table,
            "output_model": output_model,
        },
        config=config,
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
