import pandas as pd
from table_agent import extract, aextract
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel


class FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class Structure(BaseModel):
    value: str


response = "dummy"
outputs = [Structure(value="foo")]
script = """with open("output.json", "w") as f:
    f.write('[{"value": "foo"}]')
"""


# Bring any LangChain compatible model (e.g. OpenAI or Ollama)
llm = FakeChatModel(
    messages=iter(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "python_tool",
                        "args": {"script": script},
                        "id": "abc",
                    }
                ],
            ),
            # ToolMessage(
            #     content="tool response",
            #     artifact=(script, outputs),
            #     tool_call_id="abc",
            # ),
            AIMessage("dummy"),
        ]
    )
)

# Load your dataset
df = pd.DataFrame({"text": [o.value for o in outputs]})


def test_extract():
    # Let the table agent extract the data
    res = extract(df, Structure, llm=llm)

    # The model left you a kind message
    assert res["response"] == response
    assert res["script"] == script
    assert res["outputs"] == outputs
