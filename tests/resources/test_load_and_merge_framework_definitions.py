import pytest
from amlb.resources import load_and_merge_framework_definitions

framework_file = "files/frameworks.yaml"
framework_file_with_extension_only = "files/frameworks3.yaml"
second_file_has_duplicate = [
    framework_file,
    "files/frameworks2.yaml",
]
second_file_has_extension = [
    framework_file,
    framework_file_with_extension_only,
]


@pytest.mark.use_disk
def test_loads_all_definitions():
    definition = load_and_merge_framework_definitions(framework_file)
    assert "unit_test_framework" in definition
    assert len(definition) == 5


@pytest.mark.use_disk
def test_merges_definitions_of_two_files():
    definition = load_and_merge_framework_definitions(second_file_has_extension)
    assert "other_test_framework_extended_other_file" in definition
    assert len(definition) == 6


@pytest.mark.use_disk
def test_does_not_raise_exception_if_extension_is_not_defined():
    try:
        load_and_merge_framework_definitions(framework_file_with_extension_only)
    except Exception:
        pytest.fail(
            "Extensions should be verified when filling defaults, not on initial load."
        )


@pytest.mark.use_disk
def test_raises_exception_on_duplicate_definition():
    with pytest.raises(ValueError, match="Duplicate entry 'duplicate_entry' found."):
        load_and_merge_framework_definitions(second_file_has_duplicate)
