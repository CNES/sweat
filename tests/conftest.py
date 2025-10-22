import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked as slow",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")


@pytest.hookimpl(tryfirst=True)  # type: ignore[no-redef]
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
):
    """
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
    # Always run test if it's a single test invoked from vscode
    if len(items) == 1 and "vscode_pytest" in config.invocation_params.args:
        return
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    for item in items:
        # Check if all tests have a marker
        if not any(
            marker.name
            in ["unit", "functional", "end_to_end", "notebooks", "skip", "docs"]
            for marker in item.iter_markers()
        ):
            msg = (
                f"Test {item.nodeid} is missing a required marker"
                "(unit, functional, end_to_end, notebooks, slow, skip)"
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


@pytest.hookimpl(trylast=True)  # type: ignore[no-redef]
def pytest_collection_modifyitems(items: list[pytest.Item]):  # noqa
    """
    Hook to check if SWEAT_TEST_DATA_PATH variable is set
    if test with require_test_data is selected.

    Parameters
    ----------
    config: pytest.Config
        pytest configuration
    items: list[pyest.Item]
        list of pytest.Item objects (i.e., collected tests)
    """
    # Get test path directory
    test_data_path = os.getenv("SWEAT_TEST_DATA_PATH")

    for item in items:
        # Check if SWEAT_TEST_DATA_PATH is set for tests
        # with require_test_data marker
        if "require_test_data" in item.keywords and not test_data_path:
            msg = (
                f"Test {item.nodeid} is marked with @pytest.mark.require_data, "
                "but the SWEAT_TEST_DATA environment variable is not set."
            )
            raise pytest.UsageError(msg)
