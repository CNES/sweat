# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

from pathlib import Path

import nbformat
import pytest
from nbclient.exceptions import CellExecutionError, CellTimeoutError
from nbconvert.preprocessors import ExecutePreprocessor


def get_notebooks():
    return [
        path
        for path in Path("notebooks").rglob("*.ipynb")
        if (".ipynb_checkpoints") not in path.parts
        and ("Untitled" not in path.parts)
        and ("test" not in path.parts)
    ]


def get_main_notebooks():
    return list(Path("notebooks").glob("run_*.ipynb"))


@pytest.mark.notebooks
@pytest.mark.slow
@pytest.mark.require_test_data
@pytest.mark.parametrize("notebook_path", get_notebooks())
def test_notebook_execution(notebook_path):
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": notebook_path.parent}})
    except (CellExecutionError, CellTimeoutError) as e:
        pytest.fail(f"Error executing the notebook {notebook_path}:\n{e}")


@pytest.mark.notebooks
@pytest.mark.parametrize("notebook_path", get_main_notebooks())
def test_main_notebook_execution(notebook_path):
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": notebook_path.parent}})
    except (CellExecutionError, CellTimeoutError) as e:
        pytest.fail(f"Error executing the notebook {notebook_path}:\n{e}")
