import pytest
from amlb.utils import Namespace
from amlb.resources import load_framework_definitions_raw, \
    remove_frameworks_with_unknown_parent, remove_self_reference_extensions, \
    add_and_normalize_names, load_framework_definitions, from_config, \
    autocomplete_definition, autocomplete_framework_module, \
    autocomplete_framework_version, autocomplete_framework_setup_args, \
    autocomplete_setup_script, Resources, autocomplete_setup_cmd, autocomplete_params, \
    autocomplete_image, autocomplete_definition2

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

directory_aliases = [
    ("input", "my_input"),
    ("output", "my_output"),
    ("user", "my_user_dir"),
    ("root", "my_root_dir"),
]


@pytest.fixture
def simple_resource():
    return Resources(
        Namespace(
            input_dir="my_input",
            output_dir="my_output",
            user_dir="my_user_dir",
            root_dir="my_root_dir",
            docker=Namespace(
                image_defaults=Namespace(
                    author="author",
                    image=None,
                    tag=None,
                )
            ),
            frameworks=Namespace(
                root_module="frameworks",
            )
        )
    )


def test_load_framework_definition_raw_one_file():
    definition = load_framework_definitions_raw(framework_file)
    assert "unit_test_framework" in definition
    assert len(definition) == 5


def test_load_framework_definition_raw_two_files_duplicate():
    with pytest.raises(ValueError, match="Duplicate entry 'duplicate_entry' found."):
        load_framework_definitions_raw(second_file_has_duplicate)


def test_load_framework_definition_raw_two_files_extensions():
    definition = load_framework_definitions_raw(second_file_has_extension)
    assert "other_test_framework_extended_other_file" in definition
    assert len(definition) == 6


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


@pytest.mark.parametrize("name", ["a", "b"])
def test_autocomplete_framework_module_name(name):
    dummy_config = Namespace(frameworks=Namespace(root_module="frameworks"))
    framework = Namespace(name=name)
    autocomplete_framework_module(framework, dummy_config)
    assert framework.module == f"frameworks.{name}"


def test_autocomplete_framework_module_custom_module():
    dummy_config = Namespace(frameworks=Namespace(root_module="frameworks"))
    framework = Namespace(name="c", module="custom")
    autocomplete_framework_module(framework, config=dummy_config)
    assert framework.module == "custom"


def test_autocomplete_framework_module_custom_root_module():
    dummy_config = Namespace(frameworks=Namespace(root_module="different"))
    framework = Namespace(name="d")
    autocomplete_framework_module(framework, dummy_config)
    assert framework.module == "different.d"


def test_autocomplete_framework_version_set_default():
    framework = Namespace()
    autocomplete_framework_version(framework)
    assert "version" in framework
    assert framework.version == "latest"


def test_autocomplete_framework_version_specified():
    framework = Namespace(version="v1.0")
    autocomplete_framework_version(framework)
    assert "version" in framework
    assert framework.version == "v1.0"


def test_autocomplete_framework_setup_args_default_no_repo_is_version():
    f_my_version = Namespace(version="my_version")
    f_my_other_version = Namespace(version="my_other_version")

    autocomplete_framework_setup_args(f_my_version)
    autocomplete_framework_setup_args(f_my_other_version)

    assert f_my_version.setup_args == ["my_version"]
    assert f_my_other_version.setup_args == ["my_other_version"]


@pytest.mark.parametrize("version, repo", [("my_version", "my_repo"), ("my_other_version", "my_other_repo")])
def test_autocomplete_framework_setup_args_default_with_repo_is_version_then_repo(version, repo):
    f_my_version = Namespace(version=version, repo=repo)
    autocomplete_framework_setup_args(f_my_version)
    assert f_my_version.setup_args == [version, repo]


def test_autocomplete_framework_setup_args_already_set():
    f_no_extra = Namespace(setup_args="no_extra")
    autocomplete_framework_setup_args(f_no_extra)
    assert f_no_extra.setup_args == ["no_extra"]


def test_autocomplete_framework_setup_args_already_set_ignores_others():
    f_with_extra = Namespace(setup_args="w_extra", version="my_version", repo="my_repo")
    autocomplete_framework_setup_args(f_with_extra)
    assert f_with_extra.setup_args == ["w_extra"]


def test_autocomplete_setup_script_none_provided(simple_resource):
    framework = Namespace()
    autocomplete_setup_script(framework, simple_resource)
    assert framework.setup_script is None


def test_autocomplete_setup_script_static(simple_resource):
    framework = Namespace(module="my_module", setup_script="t.sh")
    autocomplete_setup_script(framework, simple_resource)
    assert framework.setup_script == "t.sh"


def test_autocomplete_setup_script_interpolates_module(simple_resource):
    framework = Namespace(module="my_module", setup_script="{module}/t.sh")
    autocomplete_setup_script(framework, simple_resource)
    assert framework.setup_script == "my_module/t.sh"


@pytest.mark.parametrize("alias, actual", directory_aliases)
def test_autocomplete_setup_script_interpolates_directory(simple_resource, alias, actual):
    framework = Namespace(setup_script=f"{{{alias}}}/t.sh", module="")
    autocomplete_setup_script(framework, simple_resource)
    assert framework.setup_script.endswith(f"{actual}/t.sh")


def test_autocomplete_setup_cmd_none_provided(simple_resource):
    framework = Namespace()
    autocomplete_setup_cmd(framework, simple_resource)
    assert framework.setup_cmd == None
    assert framework._setup_cmd == None


@pytest.mark.parametrize("commands", ["original", ["one", "two"]])
def test_autocomplete_setup_cmd_provided_original_saved(simple_resource, commands):
    framework = Namespace(setup_cmd=commands)
    autocomplete_setup_cmd(framework, simple_resource)
    assert framework._setup_cmd == commands


def test_autocomplete_setup_cmd_str_to_list(simple_resource):
    framework = Namespace(setup_cmd="str_command")
    autocomplete_setup_cmd(framework, simple_resource)
    assert framework.setup_cmd == ["str_command"]


def test_autocomplete_setup_cmd_list_unaltered(simple_resource):
    framework = Namespace(setup_cmd=["str", "commands"])
    autocomplete_setup_cmd(framework, simple_resource)
    assert framework.setup_cmd == ["str", "commands"]


@pytest.mark.parametrize("alias, actual", directory_aliases)
def test_autocomplete_setup_cmd_directory_interpolation(simple_resource, alias, actual):
    framework = Namespace(setup_cmd=[f"{{{alias}}}"])
    autocomplete_setup_cmd(framework, simple_resource)
    assert framework.setup_cmd[0].endswith(actual)


def test_autocomplete_params_none():
    framework = Namespace()
    autocomplete_params(framework)
    assert framework.params == dict()


def test_autocomplete_params_is_something():
    framework = Namespace(params=Namespace(my_param="set"))
    autocomplete_params(framework)
    assert isinstance(framework.params, dict)
    assert framework.params["my_param"] == "set"


@pytest.mark.parametrize("author", ["automlbenchmark", "unittest"])
def test_autocomplete_image_none_correct_author(simple_resource, author):
    simple_resource.config.docker.image_defaults.author = author
    framework = Namespace(version="v", name="n")
    autocomplete_image(framework, simple_resource.config)
    assert framework.image.author == author


@pytest.mark.parametrize("tag, expected", [("V1", "V1"), (None, "v0.2dev")])
def test_autocomplete_image_none_correct_tag(simple_resource, tag, expected):
    simple_resource.config.docker.image_defaults.tag = tag
    framework = Namespace(version="V0.2dev", name="n")
    autocomplete_image(framework, simple_resource.config)
    assert framework.image.tag == expected


@pytest.mark.parametrize("image, expected", [("img", "img"), (None, "decision_tree")])
def test_autocomplete_image_none_correct_image(simple_resource, image, expected):
    simple_resource.config.docker.image_defaults.image = image
    framework = Namespace(name="decision_tree", version="v0.1")
    autocomplete_image(framework, simple_resource.config)
    assert framework.image.image == expected


def test_autocomplete_image_set_author(simple_resource):
    simple_resource.config.docker.image_defaults.author = "author"
    framework = Namespace(name="n", version="0.1", image=Namespace(author="hfinley"))
    autocomplete_image(framework, simple_resource.config)
    assert framework.image.author == "hfinley"


def test_autocomplete_image_set_tag(simple_resource):
    simple_resource.config.docker.image_defaults.tag = "v1.0dev"
    framework = Namespace(name="n", image=Namespace(tag="1.0-xenial"))
    autocomplete_image(framework, simple_resource.config)
    assert framework.image.tag == "1.0-xenial"


def test_autocomplete_image_set_image(simple_resource):
    simple_resource.config.docker.image_defaults.image = "default_image"
    framework = Namespace(name="n", version="20.0a", image=Namespace(image="automl"))
    autocomplete_image(framework, simple_resource.config)
    assert framework.image.image == "automl"


def test_autocomplete_definition2_all_empty_defaults(simple_resource):
    framework_one = Namespace(name="h2o_automl")
    framework_two = Namespace(name="h2o_automl")
    autocomplete_definition(framework_one, parent=None, resource=simple_resource)
    autocomplete_definition2(framework_two, parent=None, resource=simple_resource)
    assert Namespace.dict(framework_one) == Namespace.dict(framework_two)


def test_autocomplete_definition2_all_framework_nonempty(simple_resource):
    framework_one = Namespace(name="gama", version="v0.2beta", module="custom", setup_args="password", params=Namespace(mode="best"), setup_cmd="start", setup_script="start.sh", image=Namespace(author="pgijsbers", tag="v0.2beta-xenial", image="gama"))
    framework_two = Namespace(name="gama", version="v0.2beta", module="custom", setup_args="password", params=Namespace(mode="best"), setup_cmd="start", setup_script="start.sh", image=Namespace(author="pgijsbers", tag="v0.2beta-xenial", image="gama"))
    autocomplete_definition(framework_one, parent=None, resource=simple_resource)
    autocomplete_definition2(framework_two, parent=None, resource=simple_resource)
    assert Namespace.dict(framework_one) == Namespace.dict(framework_two)


def test_autocomplete_definition2_all_both_nonempty(simple_resource):
    framework_one = Namespace(name="gama", version="v0.2beta", module="custom", setup_args="password", params=Namespace(mode="best"), setup_cmd="start", setup_script="start.sh", image=Namespace(author="pgijsbers", tag="v0.2beta-xenial", image="gama"))
    framework_two = Namespace(name="gama", version="v0.2beta", module="custom", setup_args="password", params=Namespace(mode="best"), setup_cmd="start", setup_script="start.sh", image=Namespace(author="pgijsbers", tag="v0.2beta-xenial", image="gama"))
    additional_defaults = Namespace(
        docker=Namespace(
            image_defaults=Namespace(
                author="author",
                image="default-image",
                tag="default-tag",
            )
        ),
        frameworks=Namespace(
            root_module="frameworks",
        )
    )
    simple_resource.config + additional_defaults
    autocomplete_definition(framework_one, parent=None, resource=simple_resource)
    autocomplete_definition2(framework_two, parent=None, resource=simple_resource)
    assert Namespace.dict(framework_one) == Namespace.dict(framework_two)


def test_autocomplete_definition2_all_framework_empty(simple_resource):
    framework_one = Namespace(name="gama")
    framework_two = Namespace(name="gama")
    additional_defaults = Namespace(
        docker=Namespace(
            image_defaults=Namespace(
                author="author",
                image="default-image",
                tag="default-tag",
            )
        ),
        frameworks=Namespace(
            root_module="frameworks",
        )
    )
    simple_resource.config + additional_defaults
    autocomplete_definition(framework_one, parent=None, resource=simple_resource)
    autocomplete_definition2(framework_two, parent=None, resource=simple_resource)
    assert Namespace.dict(framework_one) == Namespace.dict(framework_two)
