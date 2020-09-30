import pytest
from amlb.utils import Namespace
from amlb.resources import load_framework_definitions_raw, remove_frameworks_with_unknown_parent, remove_self_reference_extensions, add_and_normalize_names

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


def test_remove_frameworks_with_unknown_parent_removes_one():
    f = Namespace(dummy=Namespace(name="dummy", extends="does_not_exist"))
    remove_frameworks_with_unknown_parent(f)
    assert "dummy" not in f


def test_remove_frameworks_with_unknown_parent_removes_none():
    f = Namespace(
        dummy=Namespace(name="dummy", extends="does_exist"),
        does_exist=Namespace(name="does_exist"),
    )
    remove_frameworks_with_unknown_parent(f)
    assert "dummy" in f
    assert "does_exist" in f


def test_remove_self_reference_extensions_remove_one():
    f = Namespace(dummy=Namespace(name="dummy", extends="dummy"))
    remove_self_reference_extensions(f)
    assert f.dummy.extends is None


def test_remove_self_reference_extensions_remove_none():
    f = Namespace(dummy=Namespace(name="dummy", extends="some_other_framework"))
    remove_self_reference_extensions(f)
    assert f.dummy.extends is "some_other_framework"


def test_add_and_normalize_names_adds_name():
    f = Namespace(dummy=Namespace())
    add_and_normalize_names(f)
    assert "name" in f.dummy
    assert f.dummy.name == "dummy"


def test_add_and_normalize_names_name_is_normalized():
    f = Namespace(Dummy=Namespace())
    add_and_normalize_names(f)
    assert "dummy" in f, "The 'Dummy' entry should be in all lower case."
    assert "Dummy" not in f, "The old name should be invalid."
    assert f.dummy.name == "dummy"


def test_add_and_normalize_names_extension_is_normalized():
    f = Namespace(dummy=Namespace(extends="AnotherDummy"))
    add_and_normalize_names(f)
    assert f.dummy.extends == "anotherdummy"


def test_validate_framework():
    pass
    # autosklearn = Namespace(
    #     name="autosklearn",
    #     version="0.8.0",
    #     project="https://automl.github.io/auto-sklearn/"
    # )

