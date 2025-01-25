import tiktoken
from pydantic import BaseModel


def truncate(text: str, max_tokens: int, model="gpt-4o") -> str:
    """Truncates text to a maximum number of tokens

    Args:
        text (str): The text to truncate
        max_tokens (int): The maximum number of tokens
        model (str, optional): The model to use for tokenization. Defaults to "gpt-4o".
    """
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        return enc.decode(tokens[:max_tokens]) + "..."
    return text


def truncate_model(model: BaseModel, max_length: int = 10) -> BaseModel:
    """Returns a pydantic model with all string fields truncated to a maximum length

    Args:
        model (BaseModel): The model to truncate
        max_length (int, optional): The maximum length of strings. Defaults to 10.
    """
    truncated_data = {}
    for field, value in model.dict().items():
        if isinstance(value, str):
            truncated_data[field] = truncate(value, max_length)
        else:
            truncated_data[field] = value
    return model.__class__(**truncated_data)
