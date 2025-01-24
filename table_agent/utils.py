import tiktoken
from pydantic import BaseModel


def truncate(text: str, max_tokens: int, model="gpt-4o") -> str:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        return enc.decode(tokens[:max_tokens]) + "..."
    return text


def truncate_model(model: BaseModel, max_length: int = 10) -> BaseModel:
    truncated_data = {}
    for field, value in model.dict().items():
        if isinstance(value, str):
            truncated_data[field] = truncate(value, max_length)
        else:
            truncated_data[field] = value
    return model.__class__(**truncated_data)
