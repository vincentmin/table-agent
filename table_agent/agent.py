from typing import Type
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, AnyMessage, AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .types import State

SYSTEM_PROMPT = """Your job is to write a python function that extracts a single row of a pandas dataframe into a json with the following format:
{schema}

The first 5 lines of the dataframe look as follows:
{df}"""


def get_agent(
    llm: BaseChatModel,
    df: pd.DataFrame,
    output_model: Type[BaseModel],
    tools: list[BaseTool],
):
    model = llm.bind_tools(tools)

    def respond(state: State) -> AIMessage:
        system_message = SystemMessage(
            SYSTEM_PROMPT.format(df=df.head(), schema=output_model.model_json_schema())
        )
        response = model.invoke([system_message] + state["messages"])
        print(response)
        return {"messages": [response]}

    return respond
