import pytest


@pytest.mark.unit
def test_package_import_and_version() -> None:
    """Verify that callshield imports and has a valid __version__ string."""
    import callshield

    assert hasattr(callshield, "__version__")
    assert isinstance(callshield.__version__, str)
    assert len(callshield.__version__.split(".")) >= 3


@pytest.mark.unit
def test_subpackages_import() -> None:
    """Verify that all core subpackages can be imported."""
    import callshield.core
    import callshield.detectors
    import callshield.server
    import callshield.tools

    assert callshield.core is not None
    assert callshield.detectors is not None
    assert callshield.server is not None
    assert callshield.tools is not None
