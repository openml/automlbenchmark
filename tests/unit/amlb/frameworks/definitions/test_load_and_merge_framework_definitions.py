import os
import pytest
from amlb.frameworks.definitions import default_tag, _load_and_merge_framework_definitions

here = os.path.realpath(os.path.dirname(__file__))
res = os.path.join(here, 'resources')

framework_file = f"{res}/frameworks.yaml"
framework_file_with_extension_only = f"{res}/frameworks3.yaml"
second_file_has_duplicate = [
    framework_file,
    f"{res}/frameworks2.yaml",
]
second_file_has_extension = [
    framework_file,
    framework_file_with_extension_only,
]


@pytest.mark.use_disk
def test_loads_all_definitions(simple_resource):
    definitions_by_tag = _load_and_merge_framework_definitions(framework_file, simple_resource.config)
    assert len(definitions_by_tag) == 1
    definitions = definitions_by_tag[default_tag]
    assert "unit_test_framework" in definitions
    assert len(definitions) == 5


@pytest.mark.use_disk
def test_merges_definitions_of_two_files(simple_resource):
    definitions_by_tag = _load_and_merge_framework_definitions(second_file_has_extension, simple_resource.config)
    assert len(definitions_by_tag) == 1
    definitions = definitions_by_tag[default_tag]
    assert "other_test_framework_extended_other_file" in definitions
    assert len(definitions) == 6


@pytest.mark.use_disk
def test_does_not_raise_exception_if_extension_is_not_defined(simple_resource):
    try:
        _load_and_merge_framework_definitions(framework_file_with_extension_only, simple_resource.config)
    except Exception:
        pytest.fail(
            "Extensions should be verified when filling defaults, not on initial load."
        )


@pytest.mark.use_disk
def test_raises_exception_on_duplicate_definition(simple_resource):
    with pytest.raises(ValueError, match="Duplicate entry 'duplicate_entry' found."):
        _load_and_merge_framework_definitions(second_file_has_duplicate, simple_resource.config)
