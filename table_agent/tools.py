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
from .utils import truncate, truncate_model

DOCKERFILE = """FROM python:3.12-slim
RUN pip install -U pip && pip install pandas pyarrow

WORKDIR /workspace

CMD ["sleep", "infinity"]
"""


@lru_cache
def get_image():
    with tempfile.TemporaryDirectory() as tempdir:
        (Path(tempdir) / "Dockerfile").write_text(DOCKERFILE)
        return DockerImage(path=tempdir, tag="table_agent").build()


@tool(response_format="content_and_artifact")
def python_tool(
    script: Annotated[str, "The python script"],
    state: Annotated[State, InjectedState],
):
    """Run a python script.
    You can use print statements in order to see the output,
    but be aware that excessively long outputs will be truncated."""
    print("Running python script", script)
    # create temporary workspace to link to the container and run script in
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        # write the script and df to the workspace so the container can run it
        (tempdir / "script.py").write_text(script)
        state["df"].to_parquet(tempdir / "table.parquet")

        image = get_image()
        container = DockerContainer(image.tag).with_volume_mapping(
            str(tempdir), "/workspace", mode="rw"
        )
        with container as c:
            exit_code, out = c.exec("python /workspace/script.py")
            out = truncate(out.decode("utf-8"), 4000)
            text = (
                f"Your script printed the following output:\n{truncate(out.strip(), 4000)}"
                if out.strip()
                else ""
            )
            print("Output:", out)
            if exit_code != 0:
                raise ValueError(f"Error running script: {out}")

            # check if the file `output.json` was created
            output_file = tempdir / "output.json"
            if not output_file.exists():
                print("No output.json file was created")
                raise ValueError(f"{text}\n\nError: No output.json file was created")

            # read the output file
            with output_file.open() as output_file:
                outputs = json.load(output_file)

            if not isinstance(outputs, list):
                print("output.json must contain a **list** of outputs")
                raise ValueError(
                    f"{text}\n\nError: output.json must contain a **list** of outputs"
                )

            output_model = state["output_model"]

            def parse_output(idx: int, output):
                try:
                    return output_model(**output)
                except Exception as e:
                    print(f"Error validating output: {e}")
                    raise ValueError(
                        f"{text}\n\nError: Encountered an error validating line {idx} in output.json.\nError: {e}"
                    )

            outputs = [parse_output(idx, output) for idx, output in enumerate(outputs)]

            display_outputs = [truncate_model(output) for output in outputs[:5]]
            text += f"\n\nHere are the first 5 outputs:\n{display_outputs}"

            return text, (script, outputs)
