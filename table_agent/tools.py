from typing import Annotated
import json
from pathlib import Path
import tempfile
from functools import lru_cache
from testcontainers.core.container import DockerContainer
from testcontainers.core.image import DockerImage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from .types import State

DOCKERFILE = """FROM python:3.12-slim
RUN pip install -U pip && pip install pandas

WORKDIR /workspace

CMD ["sleep", "infinity"]
"""


@lru_cache
def get_image():
    with tempfile.TemporaryDirectory() as tempdir:
        with open(os.path.join(tempdir, "Dockerfile"), "w") as f:
            f.write(DOCKERFILE)
        return DockerImage(path=tempdir, tag="table_agent").build()


@tool  # (response_format="content_and_artifact")
def python_tool(
    script: Annotated[str, "The python script"],
    state: Annotated[State, InjectedState],
):
    """Run a python script. Make sure to put print statements in order to see the ouput"""
    print("Running python script", script)
    # create temporary workspace to link to the container and run script in
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        # write the script and df to the workspace so the container can run it
        (tempdir / "script.py").write_text(script)
        state["df"].to_parquet(tempdir / "table.parquet")

        image = get_image()
        container = DockerContainer(image.tag).with_volume_mapping(
            str(tempdir), "/workspace"
        )
        with container as c:
            exit_code, out = c.exec("python /workspace/script.py")
            out = out.decode("utf-8")
            print("Output:", out)
            if exit_code != 0:
                raise ValueError(f"Error running script: {out}")

            # check if the file `output.json` was created
            output_file = tempdir / "output.json"
            if not output_file.exists():
                print("No output.json file was created")
                raise ValueError("No output.json file was created")

            # read the output file
            with output_file.open() as output_file:
                outputs = json.load(output_file)

            if not isinstance(outputs, list):
                print("output.json must contain a **list** of outputs")
                raise ValueError("ouput.json must contain a **list** of outputs")

            text = (
                f"Your script printed the following output:\n{out}\n\n"
                f"Here are the first 5 outputs:\n{outputs[:5]}"
            )
            return text, outputs
