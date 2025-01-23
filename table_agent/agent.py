from typing import Type
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, AnyMessage, AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

SYSTEM_PROMPT = """Your job is to write a python function that extracts a single row of a pandas dataframe into a json with the following format:
{schema}

The first 5 lines of the dataframe look as follows:
{df}"""


def get_agent(
    llm: BaseChatModel,
    output_model: Type[BaseModel],
    df: pd.DataFrame,
    tools: list[BaseTool],
):
    model = llm.bind_tools(tools)

    def respond(messages: list[AnyMessage]) -> AIMessage:
        system_message = SystemMessage(
            SYSTEM_PROMPT.format(df=df.head(), schema=output_model.model_json_schema())
        )
        return model.invoke([system_message] + messages)

    return respond
