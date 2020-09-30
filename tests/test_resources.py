import pytest
from amlb.utils import Namespace
from amlb.resources import load_framework_definitions_raw

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


def test_load_framework_definition_raw_one_file():
    definition = load_framework_definitions_raw(framework_file)
    assert "unit_test_framework" in definition
    assert len(definition) == 4


def test_load_framework_definition_raw_two_files_duplicate():
    with pytest.raises(ValueError, match="Duplicate entry 'duplicate_entry' found."):
        load_framework_definitions_raw(second_file_has_duplicate)


def test_load_framework_definition_raw_two_files_extensions():
    definition = load_framework_definitions_raw(second_file_has_extension)
    assert "other_test_framework_extended_other_file" in definition
    assert len(definition) == 5


def test_load_framework_definition_raw_extension_no_base():
    try:
        load_framework_definitions_raw(framework_file_with_extension_only)
    except Exception:
        pytest.fail(
            "Extensions should be verified when filling defaults, not on initial load."
        )


def test_validate_framework():
    pass
    # autosklearn = Namespace(
    #     name="autosklearn",
    #     version="0.8.0",
    #     project="https://automl.github.io/auto-sklearn/"
    # )

