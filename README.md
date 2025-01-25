# table-agent

An AI agent that can efficiently parse tables (csv, parquet, ...) into a structured format.
We rely on an agent architecture to write code that parses the table programmatically.
Alternative methods rely on having an LLM inspect each row separately and extracting it into a structured format.
For large tables, these methods consume huge amounts of tokens.
For these methods, the cost and wait time scale linearly with the number of rows.
Instead, with our method the LLM inspects only the first 5 rows and then attempts to write a script that can parse all the rows.
This means we can parse milions of rows efficiently without consuming milions of tokens.


## install the package

To install the package run
```bash
git clone https://github.com/vincentmin/table-agent.git
cd table-agent
pip install .
```

If you want to run the example in `main.py`, you will need to install the extra dependencies too
```bash
pip install .[all]
python main.py
```


## extracting structured information from tables

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


## Technologies

The core technologies that the project relies on are:

- **LangChain**: Langchain allows us to be model provider agnostic. The only requirements are that the model is available through Langchain and that the model has tool calling capabilities.
- **LangGraph**: We rely on LangGraph to construct the agent in a modular and maintainable fashion.
- **Docker**: The agent writes python code that transforms the table. For safety purposes, we run the script in a docker container so the agent cannot affect your machine with malicious code.


## Tips and tricks

- The agent is capable of parsing complex tables. Nevertheless, it can be helpful if you clean the table before feeding it to the agent. For example, if you have an excel file that changes structure after a certain number of rows (i.e. the meaning of certain column changes), you may want to split that table in 2 separate tables, each with a well defined and consistent structure.
- If you know that your data has certain peculiarities, you can put this information in the user prompt.
- By default, the docker image for the container that runs the script only has `pandas` python package installed. Modify the Dockerfile if you think the agent will need additional dependencies.
- Use LangSmith or another tracing library to get full visibility into every step the agent is taking.

## Limitations

Keep in mind that the LLM will write code that parses the table programatically.
That means that the type of extractions that it can do is limited to extractions that are feasible with a script.
For example, it is best to avoid fields in the output model that require complex row-per-row reasoning.
