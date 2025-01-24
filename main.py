from typing import Literal
from langchain_openai import ChatOpenAI
from datasets import load_dataset
from pydantic import BaseModel, Field

from table_agent import extract
from table_agent.utils import truncate_model

# Load the IMDb Movie Reviews Dataset
df = load_dataset("imdb")["train"].to_pandas()

print(df.head())
#                                                 text  label
# 0  I rented I AM CURIOUS-YELLOW from my video sto...      0
# 1  "I Am Curious: Yellow" is a risible and preten...      0
# 2  If only to avoid making this type of film in t...      0
# 3  This film was probably inspired by Godard's Ma...      0
# 4  Oh, brother...after hearing about this ridicul...      0

llm = ChatOpenAI(model_name="gpt-4o-mini")


class Movie(BaseModel):
    review: str = Field(..., description="The review text")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        ..., description="The sentiment of the review"
    )


# Extract the table
res = extract(df, Movie, llm=llm)
for output in res["outputs"][:5]:
    print(truncate_model(output))
#
# review='I rented I AM CURIOUS-YELLOW from my...' sentiment='negative'
# review='"I Am Curious: Yellow" is a risible...' sentiment='negative'
# review='If only to avoid making this type of film in...' sentiment='negative'
# review="This film was probably inspired by Godard's Mascul..." sentiment='negative'
# review='Oh, brother...after hearing about this ridiculous film...' sentiment='negative'
