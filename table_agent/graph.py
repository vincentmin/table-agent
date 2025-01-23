from typing import Literal, Type
import pandas as pd
from langgraph.graph import MessageGraph
from langgraph.prebuilt import ToolNode
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from pydantic import BaseModel

from .tools import python_tool
from .agent import get_agent

type Messages = list[AnyMessage]


def get_graph(llm: BaseChatModel, df: pd.DataFrame, output_model: Type[BaseModel]):
    """Returns a langgraph runnable that extracts data from a table

    Args:
        llm (BaseChatModel): A language model
        df (pd.DataFrame): The table to be parsed
        output_model (Type[BaseModel]): The table will be parsed into a list of instances of this output model
    """
    tools = [python_tool]
    tool_node = ToolNode(tools)
    agent = get_agent(llm, df, output_model, tools)

    def route(messages: Messages) -> Literal["tools", "__end__"]:
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    builder = MessageGraph()
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)
    builder.add_edge("__start__", "agent")
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges("agent", route)

    return builder.compile()
