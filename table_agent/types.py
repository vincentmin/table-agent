from typing import Type
import pandas as pd
from pydantic import BaseModel
from langgraph.graph import MessagesState


class State(MessagesState):
    df: pd.DataFrame
    output_model: Type[BaseModel]
    docker_image: str
