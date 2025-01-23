from typing import Type
import pandas as pd
from langchain_core.language_models import BaseChatModel

type TableInput = pd.DataFrame

def load_default_model() -> BaseChatModel:
    """Loads a default model

    Returns:
        BaseChatModel: A default model
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model_name="gpt-4o-mini")

def extract[T](table: TableInput, output_model: Type[T], llm: BaseChatModel | None = None) -> list[T]:
    """Takes in a table and an output model and returns a list of output models

    Args:
        table (TableInput): A pandas DataFrame
        output_model (Type[T]): A class that represents the output model

    Returns:
        list[T]: A list of output models
    """
    if llm is None:
        llm = load_default_model()
    pass
