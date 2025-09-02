# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import subprocess

import pytest


@pytest.mark.docs
@pytest.mark.slow
def test_mkdocs_build(tmp_path):
    """Test that mkdocs builds the documentation successfully."""
    # Optional: If mkdocs.yml is in a different directory, change `cwd`
    result = subprocess.run(
        ["mkdocs", "build", "--site-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"MkDocs build failed:\n{result.stderr}"
