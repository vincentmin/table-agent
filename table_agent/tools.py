from typing import Annotated
import os
import tempfile
from functools import lru_cache
from testcontainers.core.container import DockerContainer
from testcontainers.core.image import DockerImage
from langchain_core.tools import tool

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
def python_tool(script: Annotated[str, "The python script"]):
    """Run a python script. Make sure to put print statements in order to see the ouput"""
    with tempfile.TemporaryDirectory() as tempdir:
        with open(os.path.join(tempdir, "script.py"), "w") as f:
            f.write(script)

        image = get_image()
        container = DockerContainer(image.tag).with_volume_mapping(
            tempdir, "/workspace"
        )

        with container as c:
            exit_code, out = c.exec("python /workspace/script.py")
            out = out.decode("utf-8")
            if exit_code != 0:
                raise ValueError(f"Error running script: {out}")
            return out
