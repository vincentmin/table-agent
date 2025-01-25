from typing import Literal
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.utils.runnable import RunnableCallable
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

from .types import State


def get_graph(llm: BaseChatModel, tools: list[BaseTool] | None):
    """Build a langgraph agent with tools.

    Args:
        llm (BaseChatModel): A language model with tool calling capabilities.
        tools (list[BaseTool] | None): The tools the agent has access to.
    """
    tool_node = ToolNode(tools)
    model = llm.bind_tools(tools)

    def agent(state: State, config: RunnableConfig) -> AIMessage:
        """Synchronously invoke the language model. Used with graph.invoke."""
        response = model.invoke(state["messages"], config)
        return {"messages": [response]}

    async def aagent(state: State, config: RunnableConfig) -> AIMessage:
        """Asynchronously invoke the language model. Used with graph.ainvoke."""
        response = await model.ainvoke(state["messages"], config)
        return {"messages": [response]}

    def route(state: State) -> Literal["tools", "__end__"]:
        """Route to the tools node if the agent want to call a tool or end the conversation."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    builder = StateGraph(State)
    builder.add_node("agent", RunnableCallable(agent, aagent))
    builder.add_node("tools", tool_node)
    builder.add_edge("__start__", "agent")
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges("agent", route)

    return builder.compile()
