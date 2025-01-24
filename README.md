# table-agent

An AI agent that can efficiently parse tables (csv, parquet, ...) into a structured format.
We rely on an agent architecture to write code that parses the table programmatically.
Alternative methods rely on having an LLM inspect each row separately and extracting it into a structured format.
For large tables, these methods consume huge amounts of tokens.
For these methods, the cost and wait time scale linearly with the number of rows.
Instead, with our method the LLM inspects only the first 5 rows and then attempts to write a script that can parse all the rows.
This means we can parse milions of rows efficiently without consuming milions of tokens.

## Getting started

See `main.py` for a full example.

The main entrypoint is the `extract` method:
```python
from typing import Literal
from pydantic import BaseModel, Field

from table_agent import extract

# Define the structure you want as output using Pydantic
class Movie(BaseModel):
    review: str = Field(..., description="The review text")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        ..., description="The sentiment of the review"
    )


# Let the table agent extract the data
res = extract(df, Movie)

# Inspect your output
for output in res["outputs"][:5]:
    print(truncate_model(output))
# review='I rented I AM CURIOUS-YELLOW from my...' sentiment='negative'
# review='"I Am Curious: Yellow" is a risible...' sentiment='negative'
# review='If only to avoid making this type of film in...' sentiment='negative'
# review="This film was probably inspired by Godard's Mascul..." sentiment='negative'
# review='Oh, brother...after hearing about this ridiculous film...' sentiment='negative'
```
