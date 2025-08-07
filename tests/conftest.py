import os

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
):
    """
    Description
    -----------
    Hook to modify collected test items.

    Parameters
    ----------
    config: pytest.Config
        pytest configuration
    items: list[pyest.Item]
        list of pytest.Item objects (i.e., collected tests)
    """
    selected_items = []
    deselected_items = []
    # Get test path directory
    test_data_path = os.getenv("EVASPA_TEST_DATA_PATH")
    # Always run test if it's a single test invoked from vscode
    if len(items) == 1 and "vscode_pytest" in config.invocation_params.args:
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
    for item in items:
        # Check if all tests have a marker
        if not any(
            marker.name
            in ["unit", "functional", "end_to_end", "notebooks", "skip"]
            for marker in item.iter_markers()
        ):
            msg = f"Test {item.nodeid} is missing a required marker (unit, functional, end_to_end, notebooks, skip)"
            raise pytest.UsageError(msg)
        # Check if EVASPA_TEST_DATA_PATH is set for tests with require_test_data marker
        if "require_test_data" in item.keywords and not test_data_path:
            msg = (
                f"Test {item.nodeid} is marked with @pytest.mark.require_data, "
                "but the HAS_DATA environment variable is not set."
            )
            raise pytest.UsageError(msg)
        # Skip slow in VSCode
        if (
            "slow" in item.keywords
            and "vscode_pytest" in config.invocation_params.args
        ):
            deselected_items.append(item)
            continue
        selected_items.append(item)
    # Modify items in-place
    items[:] = selected_items

    # Inform pytest of deselected items
    config.hook.pytest_deselected(items=deselected_items)
