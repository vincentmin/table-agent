from typing import Type
import pandas as pd
from pydantic import BaseModel
from langgraph.graph import MessagesState


class State[T: BaseModel](MessagesState):
    df: pd.DataFrame
    output_model: Type[T]
    docker_image: str
