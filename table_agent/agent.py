from typing import Type
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .types import State

SYSTEM_PROMPT = """Your job is to extract data from a table by writing a python script.
The python script must read the table as a pandas dataframe from `table.parquet`.
Then write code that parses each row of the dataframe into a json with the following format:
{schema}
The script should write the parsed jsons to a file called `output.json`.
Make sure that `ouput.json` is a list of jsons that comply with the above format.

Here follow the first 5 lines of the dataframe you need to act on:
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
