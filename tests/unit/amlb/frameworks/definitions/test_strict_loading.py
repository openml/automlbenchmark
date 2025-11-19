import os
import pytest
import tempfile
from amlb.frameworks.definitions import load_framework_definitions

here = os.path.realpath(os.path.dirname(__file__))
res = os.path.join(here, "resources")


@pytest.mark.use_disk
def test_missing_framework_file_raises_error(simple_resource):
    """Missing base framework file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_framework_definitions(
            "/tmp/nonexistent_frameworks.yaml", simple_resource.config
        )

    error_message = str(exc_info.value)
    assert "not found" in error_message.lower()
    assert "typo" in error_message.lower()


@pytest.mark.use_disk
def test_missing_tagged_framework_file_is_tolerated(simple_resource):
    """Missing tagged variant files (e.g., frameworks_stable.yaml) should be tolerated."""
    # Create a minimal framework file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
TestFramework:
  version: "1.0"
""")
        temp_file = f.name

    try:
        # This should work even if tagged variants don't exist
        # (e.g., frameworks_stable.yaml, frameworks_latest.yaml, etc.)
        definitions_by_tag = load_framework_definitions(
            temp_file, simple_resource.config
        )

        # Should have loaded at least the default tag
        assert len(definitions_by_tag) >= 1
    finally:
        os.unlink(temp_file)


@pytest.mark.use_disk
def test_multiple_framework_files_first_must_exist(simple_resource):
    """When loading multiple files, all base files must exist."""
    existing_file = f"{res}/frameworks_inheritance.yaml"
    nonexistent_file = "/tmp/nonexistent_frameworks.yaml"

    # Both files in the list must exist at the base level
    with pytest.raises(FileNotFoundError):
        load_framework_definitions(
            [existing_file, nonexistent_file], simple_resource.config
        )
