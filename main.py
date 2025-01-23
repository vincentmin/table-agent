from typing import Literal
from langchain_ollama import ChatOllama
from datasets import load_dataset
from pydantic import BaseModel, Field

from table_agent import extract

# Load the IMDb Movie Reviews Dataset
df = load_dataset("imdb")["train"].to_pandas()

print(df.head())
#                                                 text  label
# 0  I rented I AM CURIOUS-YELLOW from my video sto...      0
# 1  "I Am Curious: Yellow" is a risible and preten...      0
# 2  If only to avoid making this type of film in t...      0
# 3  This film was probably inspired by Godard's Ma...      0
# 4  Oh, brother...after hearing about this ridicul...      0

llm = ChatOllama(model="llama3.2:1b")


class Movie(BaseModel):
    review: str = Field(..., description="The review text")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        ..., description="The sentiment of the review"
    )


# Extract the table
outputs = extract(df, Movie, llm=llm)
for output in outputs[:5]:
    print(output)
