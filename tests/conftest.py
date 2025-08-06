import os
from typing import Any

import pytest


def pytest_collection_modifyitems(items: Any):
    # Check if all tests have a marker
    for item in items:
        if not any(
            marker.name
            in ["unit", "functional", "end_to_end", "notebooks", "skip"]
            for marker in item.iter_markers()
        ):
            msg = f"Test {item.nodeid} is missing a required marker (unit, functional, end_to_end, notebooks, skip)"
            raise pytest.UsageError(msg)
    # Check if EVASPA_TEST_DATA_PATH is set for tests with require_test_data marker
    test_data_path = os.getenv("EVASPA_TEST_DATA_PATH")
    for item in items:
        if "require_test_data" in item.keywords and not test_data_path:
            msg = (
                f"Test {item.nodeid} is marked with @pytest.mark.require_data, "
                "but the HAS_DATA environment variable is not set."
            )
            raise pytest.UsageError(msg)
