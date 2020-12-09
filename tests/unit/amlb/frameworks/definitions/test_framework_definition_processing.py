import pytest
from amlb.utils import Namespace
from amlb.frameworks.definitions import _sanitize_and_add_defaults, \
    _add_framework_name, _find_all_parents, \
    _update_frameworks_with_parent_definitions, _remove_self_reference_extensions, \
    _remove_frameworks_with_unknown_parent


def test_remove_frameworks_with_unknown_parent_removes_framework_with_unknown_parent():
    f = Namespace(dummy=Namespace(name="dummy", extends="does_not_exist"))
    _remove_frameworks_with_unknown_parent(f)
    assert "dummy" not in f


def test_remove_frameworks_with_unknown_parent_keeps_children_with_known_parents():
    f = Namespace(
        dummy=Namespace(name="dummy", extends="does_exist"),
        does_exist=Namespace(name="does_exist"),
    )
    _remove_frameworks_with_unknown_parent(f)
    assert "dummy" in f
    assert "does_exist" in f


def test_remove_self_reference_extensions_removes_self_reference():
    f = Namespace(dummy=Namespace(name="dummy", extends="dummy"))
    _remove_self_reference_extensions(f)
    assert f.dummy.extends is None


def test_remove_self_reference_extensions_does_not_remove_reference_to_other():
    f = Namespace(dummy=Namespace(name="dummy", extends="some_other_framework"))
    _remove_self_reference_extensions(f)
    assert f.dummy.extends is "some_other_framework"


def test_add_framework_name_adds_name_attribute_to_framework_definition():
    f = Namespace(dummy=Namespace())
    _add_framework_name(f)
    assert "name" in f.dummy
    assert f.dummy.name == "dummy"


def test_add_framework_name_does_not_change_name_case():
    f = Namespace(Dummy=Namespace())
    _add_framework_name(f)
    assert "Dummy" in f
    assert f.Dummy.name == "Dummy"


def test_find_all_parents_returns_empty_list_if_framework_has_no_parent():
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        h2o_automl=Namespace(name="h2o", version="1.3"),
    )
    parents = _find_all_parents(frameworks.gama, frameworks)
    assert parents == []


def test_find_all_parents_ignores_when_extends_is_none():
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", extends=None),
    )
    assert _find_all_parents(frameworks.gama, frameworks) == [], """
            Extends `None` should behave as if it does not extend any definition.
            Extends `None` is used when an unknown extension is removed.
    """



@pytest.mark.parametrize("framework", ["gama", "h2o_automl"])
def test_find_all_parents_returns_parent_of_framework_with_single_parent(framework):
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        gama_old=Namespace(name="gama_20.1.0", version="20.1.0", extends="gama"),
        h2o_automl=Namespace(name="h2o", version="latest"),
        h2o_automl_old=Namespace(name="h2o_1.2", version="1.2", extends="h2o_automl"),
    )
    parents = _find_all_parents(frameworks[f"{framework}_old"], frameworks)
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
    parents = _find_all_parents(frameworks[f"{framework}_older"], frameworks)
    assert parents == [frameworks[f"{framework}_old"], frameworks[framework]]


@pytest.mark.parametrize(
    "field, value",
    [
        ("description", "flexible automl"),
        ("params", dict(foo="bar")),
        ("version", "20.1.0"),
        ("setup_args", "zahradnik"),
    ]
)
def test_update_frameworks_with_parent_definitions_add_missing_field_from_parent(field, value):
    frameworks = Namespace(
        gama=Namespace(name="gama", **{field: value}),
        gama_old=Namespace(name="gama_20.1.0", extends="gama"),
    )
    _update_frameworks_with_parent_definitions(frameworks)
    assert frameworks.gama_old[field] == value


@pytest.mark.parametrize(
    "field, p_value, c_value",
    [
        ("description", "flexible automl", "just automl"),
        ("params", dict(foo="bar"), dict(foo="baz")),
        ("params", dict(foo="bar"), dict(bar="baz")),
        ("version", "20.1.0", "20.1"),
        ("setup_args", "zahradnik", "smith"),
    ]
)
def test_update_frameworks_with_parent_definitions_does_not_overwrite_child(field, p_value, c_value):
    frameworks = Namespace(
        gama=Namespace(name="gama", **{field: p_value}),
        gama_old=Namespace(name="gama_20.1", **{field: c_value}, extends="gama"),
    )
    _update_frameworks_with_parent_definitions(frameworks)
    assert frameworks.gama_old[field] == c_value


@pytest.mark.parametrize(
    "field, g_value, p_value",
    [
        ("description", "flexible automl", "just automl"),
        ("params", dict(foo="bar"), dict(foo="baz")),
        ("params", dict(foo="bar"), dict(bar="baz")),
        ("version", "20.1.0", "20.1"),
        ("setup_args", "zahradnik", "smith"),
    ]
)
def test_update_frameworks_with_parent_definitions_parent_overwrites_grandparent_yaml(field, g_value, p_value):
    frameworks = Namespace(
        gama=Namespace(name="gama", **{field: g_value}),
        gama_old=Namespace(name="gama_2", **{field: p_value}, extends="gama"),
        gama_oldest=Namespace(name="gama_1", extends="gama_old"),
    )
    _update_frameworks_with_parent_definitions(frameworks)
    assert frameworks.gama_oldest[field] == p_value


def test_sanitize_and_add_defaults_root_definition_get_module(simple_resource):
    frameworks = Namespace(auto_sklearn=Namespace())
    _sanitize_and_add_defaults(frameworks, simple_resource.config)
    assert frameworks.auto_sklearn.module == "frameworks.auto_sklearn"


def test_sanitize_and_add_defaults_child_inherits_module(simple_resource):
    frameworks = Namespace(
        auto_sklearn=Namespace(),
        auto_sklearn_old=Namespace(extends="auto_sklearn")
    )
    _sanitize_and_add_defaults(frameworks, simple_resource.config)
    assert frameworks.auto_sklearn_old.module == "frameworks.auto_sklearn"


def test_sanitize_and_add_defaults_defined_module_not_overwritten(simple_resource):
    frameworks = Namespace(auto_sklearn=Namespace(module="custom_module"))
    _sanitize_and_add_defaults(frameworks, simple_resource.config)
    assert frameworks.auto_sklearn.module == "custom_module"
