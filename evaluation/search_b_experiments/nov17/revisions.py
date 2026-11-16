"""Load a version of a module from the current Git repository."""

import importlib.util
import subprocess
import tempfile
from pathlib import Path


def from_git(revision: str, source: str):
    contents = subprocess.check_output(["git", "show", f"{revision}:{source}"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / Path(source).name
        path.write_bytes(contents)
        spec = importlib.util.spec_from_file_location(f"trial_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
