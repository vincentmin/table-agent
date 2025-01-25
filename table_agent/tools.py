from typing import Annotated
import json
from pathlib import Path
import tempfile
from testcontainers.core.container import DockerContainer
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from .types import State
from .utils import truncate, truncate_model


@tool(response_format="content_and_artifact")
def python_tool(
    script: Annotated[str, "The python script"],
    state: Annotated[State, InjectedState],
):
    """Run a python script.
    You can use print statements in order to see the output,
    but be aware that excessively long outputs will be truncated."""
    # create temporary workspace to link to the container and run script in
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        # write the script and df to the workspace so the container can run it
        (tempdir / "script.py").write_text(script)
        state["df"].to_parquet(tempdir / "table.parquet")

        container = DockerContainer(state["docker_image"]).with_volume_mapping(
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

            # check if the script ran successfully
            if exit_code != 0:
                raise ValueError(f"Error running script: {out}")

            # check if the file `output.json` was created
            output_file = tempdir / "output.json"
            if not output_file.exists():
                raise ValueError(f"{text}\n\nError: No output.json file was created")

            # read the output file
            with output_file.open() as output_file:
                outputs = json.load(output_file)

            # check if the output is a list
            if not isinstance(outputs, list):
                raise ValueError(
                    f"{text}\n\nError: output.json must contain a **list** of outputs"
                )

            output_model = state["output_model"]

            # check if each item can be parsed into the output model
            def parse_output(idx: int, output):
                try:
                    return output_model(**output)
                except Exception as e:
                    raise ValueError(
                        f"{text}\n\nError: Encountered an error validating line {idx} in output.json.\nError: {e}"
                    )

            outputs = [parse_output(idx, output) for idx, output in enumerate(outputs)]

            # craft the response to the LLM
            display_outputs = [truncate_model(output) for output in outputs[:5]]
            text += f"\n\nHere are the first 5 outputs:\n{display_outputs}"

            return text, (script, outputs)
