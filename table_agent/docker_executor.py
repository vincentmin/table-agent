from pathlib import Path
import tempfile
from testcontainers.core.image import DockerImage

DOCKERFILE = """FROM python:3.12-slim
RUN pip install -U pip && pip install pandas pyarrow

WORKDIR /workspace

CMD ["sleep", "infinity"]
"""


def get_image(dockerfile: str | None = None) -> DockerImage:
    """Build a DockerImage from a Dockerfile string.

    Args:
        dockerfile (str, optional): The Dockerfile string. Defaults to None.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        (Path(tempdir) / "Dockerfile").write_text(dockerfile or DOCKERFILE)
        return DockerImage(path=tempdir, tag="table_agent").build()
