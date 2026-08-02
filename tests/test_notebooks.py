# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

import os
from pathlib import Path

import nbformat
import pytest
from nbclient.exceptions import CellExecutionError, CellTimeoutError
from nbconvert.preprocessors import ExecutePreprocessor


def get_notebooks():
    """
    Get notebooks list
    """
    return [
        path
        for path in Path("notebooks").rglob("*.ipynb")
        if (
            (".ipynb_checkpoints") not in path.parts
            and ("Untitled" not in path.name)
            and ("test" not in path.name)
            and ("validation_" not in path.name)
            and ("prepare_" not in path.name)
        )
    ]


def get_main_notebooks():
    """
    Get main notebooks list
    """
    return list(Path("notebooks").glob("run_*.ipynb"))


@pytest.mark.notebooks
@pytest.mark.slow
@pytest.mark.require_test_data
@pytest.mark.parametrize(
    "notebook_path", get_notebooks(), ids=lambda path: path.stem
)
def test_notebook_execution(notebook_path):
    """
    Test run notebooks
    """
    os.environ["NB_CONVERT"] = "1"
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": notebook_path.parent}})
    except CellExecutionError as e:
        pytest.fail(f"Error executing the notebook {notebook_path}:\n{e}")
    except CellTimeoutError as e:
        pytest.fail(f"Timeout executing the notebook {notebook_path}:\n{e}")


@pytest.mark.notebooks
@pytest.mark.require_test_data
@pytest.mark.parametrize("version", ["1.3", "1.4"])
def test_stic_notebook_execution(version):
    """
    Test run main notebooks with different versions
    """
    notebook_path = Path("notebooks") / "stic" / "model_validation.ipynb"
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Find and replace the version cell
    for cell in nb.cells:
        if 'version = "1.3"' in cell.source:
            cell.source = f'version = "{version}"'
            break

    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": notebook_path.parent}})
    except CellExecutionError as e:
        pytest.fail(
            f"Error executing the STIC notebook (version {version}):\n{e}"
        )
    except CellTimeoutError as e:
        pytest.fail(
            f"Timeout executing the STIC notebook (version {version}):\n{e}"
        )


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "notebook_path", get_main_notebooks(), ids=lambda path: path.stem
)
def test_main_notebook_execution(notebook_path):
    """
    Test run main notebooks
    """
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": notebook_path.parent}})
    except CellExecutionError as e:
        pytest.fail(f"Error executing the notebook {notebook_path}:\n{e}")
    except CellTimeoutError as e:
        pytest.fail(f"Timeout executing the notebook {notebook_path}:\n{e}")
