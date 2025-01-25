from typing import Literal
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from .tools import python_tool
from .types import State


def get_graph(llm: BaseChatModel, additional_tools: list[BaseTool] | None):
    """Returns a langgraph runnable that extracts data from a table

    Args:
        llm (BaseChatModel): A language model
        additional_tools (list[BaseTool] | None): Any additional tools to be added to the standard tools
    """
    tools = additional_tools or []
    tools.append(python_tool)
    tool_node = ToolNode(tools)
    model = llm.bind_tools(tools)

    def agent(state: State) -> AIMessage:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def route(state: State) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)
    builder.add_edge("__start__", "agent")
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges("agent", route)

    return builder.compile()
