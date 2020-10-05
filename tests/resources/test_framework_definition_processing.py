import pytest
from amlb.utils import Namespace
from amlb.resources import remove_frameworks_with_unknown_parent, \
    remove_self_reference_extensions, \
    add_and_normalize_names,\
    update_frameworks_with_parent_definitions, find_all_parents, \
    sanitize_and_add_defaults


def test_remove_frameworks_with_unknown_parent_removes_framework_with_unknown_parent():
    f = Namespace(dummy=Namespace(name="dummy", extends="does_not_exist"))
    remove_frameworks_with_unknown_parent(f)
    assert "dummy" not in f


def test_remove_frameworks_with_unknown_parent_keeps_children_with_known_parents():
    f = Namespace(
        dummy=Namespace(name="dummy", extends="does_exist"),
        does_exist=Namespace(name="does_exist"),
    )
    remove_frameworks_with_unknown_parent(f)
    assert "dummy" in f
    assert "does_exist" in f


def test_remove_self_reference_extensions_removes_self_reference():
    f = Namespace(dummy=Namespace(name="dummy", extends="dummy"))
    remove_self_reference_extensions(f)
    assert f.dummy.extends is None


def test_remove_self_reference_extensions_does_not_remove_reference_to_other():
    f = Namespace(dummy=Namespace(name="dummy", extends="some_other_framework"))
    remove_self_reference_extensions(f)
    assert f.dummy.extends is "some_other_framework"


def test_add_and_normalize_names_adds_name():
    f = Namespace(dummy=Namespace())
    add_and_normalize_names(f)
    assert "name" in f.dummy
    assert f.dummy.name == "dummy"


def test_add_and_normalize_names_converts_name_to_lower_case():
    f = Namespace(Dummy=Namespace())
    add_and_normalize_names(f)
    assert "dummy" in f, "The 'Dummy' entry should be in all lower case."
    assert f.dummy.name == "dummy"


def test_add_and_normalize_names_original_removed_for_normalized_framework():
    f = Namespace(Dummy=Namespace())
    add_and_normalize_names(f)
    assert "Dummy" not in f, "The old name should be invalid."


def test_add_and_normalize_names_extension_is_normalized():
    f = Namespace(dummy=Namespace(extends="AnotherDummy"))
    add_and_normalize_names(f)
    assert f.dummy.extends == "anotherdummy"


def test_find_all_parents_returns_empty_list_if_framework_has_no_parent():
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        h2o_automl=Namespace(name="h2o", version="1.3"),
    )
    parents = find_all_parents(frameworks.gama, frameworks)
    assert parents == []


@pytest.mark.parametrize("framework", ["gama", "h2o_automl"])
def test_find_all_parents_returns_parent_of_framework_with_single_parent(framework):
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        gama_old=Namespace(name="gama_20.1.0", version="20.1.0", extends="gama"),
        h2o_automl=Namespace(name="h2o", version="latest"),
        h2o_automl_old=Namespace(name="h2o_1.2", version="1.2", extends="h2o_automl"),
    )
    parents = find_all_parents(frameworks[f"{framework}_old"], frameworks)
    assert parents == [frameworks[framework]]


@pytest.mark.parametrize("framework", ["gama", "h2o_automl"])
def test_find_all_parents_returns_frameworks_closest_first_if_two_parents(framework):
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        gama_old=Namespace(name="gama_20.1.0", version="20.1.0", extends="gama"),
        gama_older=Namespace(name="gama_20.0.0", version="20.0.0", extends="gama_old"),
        h2o_automl=Namespace(name="h2o", version="latest"),
        h2o_automl_old=Namespace(name="h2o_1.2", version="1.2", extends="h2o_automl"),
        h2o_automl_older=Namespace(name="h2o_1.1", version="1.1", extends="h2o_automl_old"),
    )
    parents = find_all_parents(frameworks[f"{framework}_older"], frameworks)
    assert parents == [frameworks[f"{framework}_old"], frameworks[framework]]


def test_update_frameworks_with_parent_definitions_add_missing_field_from_parent():
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        gama_old=Namespace(name="gama_20.1.0", version="20.1.0", extends="gama"),
    )
    update_frameworks_with_parent_definitions(frameworks)
    assert frameworks.gama_old.description == "flexible automl"


def test_update_frameworks_with_parent_definitions_does_not_overwrite_child_yaml():
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        gama_old=Namespace(name="gama_20.1", version="20.1", description="old gama", extends="gama"),
    )
    update_frameworks_with_parent_definitions(frameworks)
    assert frameworks.gama_old.description == "old gama"


def test_update_frameworks_with_parent_definitions_parent_overwrites_grandparent_yaml():
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        gama_old=Namespace(name="gama_2", version="1", description="automl", extends="gama"),
        gama_oldest=Namespace(name="gama_1", version="2", extends="gama_old"),
    )
    update_frameworks_with_parent_definitions(frameworks)
    assert frameworks.gama_oldest.description == "automl"


def test_sanitize_and_add_defaults_root_definition_get_module(simple_resource):
    frameworks = Namespace(auto_sklearn=Namespace())
    sanitize_and_add_defaults(frameworks, simple_resource)
    assert frameworks.auto_sklearn.module == "frameworks.auto_sklearn"


def test_sanitize_and_add_defaults_child_inherits_module(simple_resource):
    frameworks = Namespace(
        auto_sklearn=Namespace(),
        auto_sklearn_old=Namespace(extends="auto_sklearn")
    )
    sanitize_and_add_defaults(frameworks, simple_resource)
    assert frameworks.auto_sklearn_old.module == "frameworks.auto_sklearn"


def test_sanitize_and_add_defaults_defined_module_not_overwritten(simple_resource):
    frameworks = Namespace(auto_sklearn=Namespace(module="custom_module"))
    sanitize_and_add_defaults(frameworks, simple_resource)
    assert frameworks.auto_sklearn.module == "custom_module"
