import pytest


@pytest.mark.unit
def test_all_custom_markers_registered(pytestconfig: pytest.Config) -> None:
    """Verify that all required custom markers are registered in pytest configuration."""
    registered_markers = pytestconfig.getini("markers")
    marker_names = [m.split(":")[0].strip() for m in registered_markers]

    expected_markers = ["unit", "integration", "e2e", "slow", "gpu"]
    for marker in expected_markers:
        assert marker in marker_names, (
            f"Expected marker '{marker}' to be registered in pytest.ini_options"
        )


@pytest.mark.unit
def test_unit_marker_dummy() -> None:
    pass


@pytest.mark.integration
def test_integration_marker_dummy() -> None:
    pass


@pytest.mark.e2e
def test_e2e_marker_dummy() -> None:
    pass


@pytest.mark.slow
def test_slow_marker_dummy() -> None:
    pass


@pytest.mark.gpu
def test_gpu_marker_dummy() -> None:
    pass
