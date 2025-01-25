from typing import Type, TypedDict
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from .graph import get_graph
from .types import State
from .docker_executor import get_image
from .tools import python_tool


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
Make sure that `output.json` contains a list of jsons that comply with the above format.

Here follow the first 5 lines of the dataframe you need to act on:
{df}"""


def extract[T: BaseModel](
    table: pd.DataFrame,
    output_model: Type[T],
    llm: BaseChatModel | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    docker_image_tag: str | None = None,
    tools: list[BaseTool] | None = None,
    config: RunnableConfig | None = None,
) -> TableOutput[T]:
        """Takes in a table and an output model and returns a list of output models

    Args:
        table (pd.DataFrame): A pandas DataFrame.
        output_model (Type[T]): A pydantic model that the table will be parsed into.
        llm (BaseChatModel, optional): A language model with tool calling capabilities.
            Defaults to None in which case OpenAI gpt-4o-mini is used.
        system_prompt (str, optional): The system prompt.
            Defaults to None in which case the default system prompt is used.
            Base yourself on the default system prompt to create your own.
            We strongly advise to include the schema and a truncated version of your table.
        user_prompt (str, optional): The user prompt.
            Defaults to None in which case "Extract data from table" is used.
        docker_image_tag (str, optional): The tag of the docker image to run llm-generated
            scripts in. Defaults to None, which uses a python-3.12 image with pandas and
            pyarrow installed. Provide a custom image tag if you need additional dependencies.
        tools (list[BaseTool], optional): The tools that the agent can use to complete its task.
            Defaults to None, in which case the python_tool is used. If you provide a list
            of tools, make sure it includes the python_tool or a substitute.
        config (RunnableConfig, optional): Langchain configuration. Defaults to None.
            See langchain for documentation. If you run into recursion errors, you may want to
            set the config to `{"recursion_limit": some_high_int}`.

    Returns:
        TableOutput: The model response, the script, and the outputs
    """

    llm = llm or load_default_model()
    system_prompt = system_prompt or SYSTEM_PROMPT.format(
        df=table.head(), schema=output_model.model_json_schema()
    )
    user_prompt = user_prompt or "Extract data from table"
    docker_image_tag = docker_image_tag or get_image().tag
    tools = tools or [python_tool]

    graph = get_graph(llm, tools)
    state: State = graph.invoke(
        {
            "messages": [SystemMessage(system_prompt), HumanMessage(user_prompt)],
            "df": table,
            "output_model": output_model,
            "docker_image": docker_image_tag,
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


async def aextract[T: BaseModel](
    table: pd.DataFrame,
    output_model: Type[T],
    llm: BaseChatModel | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    docker_image_tag: str | None = None,
    tools: list[BaseTool] | None = None,
    config: RunnableConfig | None = None,
) -> TableOutput[T]:
    """Takes in a table and an output model and returns a list of output models

    Args:
        table (pd.DataFrame): A pandas DataFrame.
        output_model (Type[T]): A pydantic model that the table will be parsed into.
        llm (BaseChatModel, optional): A language model with tool calling capabilities.
            Defaults to None in which case OpenAI gpt-4o-mini is used.
        system_prompt (str, optional): The system prompt.
            Defaults to None in which case the default system prompt is used.
            Base yourself on the default system prompt to create your own.
            We strongly advise to include the schema and a truncated version of your table.
        user_prompt (str, optional): The user prompt.
            Defaults to None in which case "Extract data from table" is used.
        docker_image_tag (str, optional): The tag of the docker image to run llm-generated
            scripts in. Defaults to None, which uses a python-3.12 image with pandas and
            pyarrow installed. Provide a custom image tag if you need additional dependencies.
        tools (list[BaseTool], optional): The tools that the agent can use to complete its task.
            Defaults to None, in which case the python_tool is used. If you provide a list
            of tools, make sure it includes the python_tool or a substitute.
        config (RunnableConfig, optional): Langchain configuration. Defaults to None.
            See langchain for documentation. If you run into recursion errors, you may want to
            set the config to `{"recursion_limit": some_high_int}`.

    Returns:
        TableOutput: The model response, the script, and the outputs
    """

    llm = llm or load_default_model()
    system_prompt = system_prompt or SYSTEM_PROMPT.format(
        df=table.head(), schema=output_model.model_json_schema()
    )
    user_prompt = user_prompt or "Extract data from table"
    docker_image_tag = docker_image_tag or get_image().tag
    tools = tools or [python_tool]

    graph = get_graph(llm, tools)
    state: State = await graph.ainvoke(
        {
            "messages": [SystemMessage(system_prompt), HumanMessage(user_prompt)],
            "df": table,
            "output_model": output_model,
            "docker_image": docker_image_tag,
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
