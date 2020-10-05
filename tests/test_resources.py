import pytest
from amlb.utils import Namespace
from amlb.resources import load_and_merge_framework_definitions, \
    remove_frameworks_with_unknown_parent, remove_self_reference_extensions, \
    add_and_normalize_names, load_framework_definitions, from_config, \
    autocomplete_definition, autocomplete_framework_module, \
    autocomplete_framework_version, autocomplete_framework_setup_args, \
    autocomplete_setup_script, Resources, autocomplete_setup_cmd, autocomplete_params, \
    autocomplete_image, add_defaults_to_frameworks, \
    update_frameworks_with_parent_definitions, find_all_parents, \
    sanitize_and_add_defaults

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


def test_find_all_parents_no_parent():
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        h2o_automl=Namespace(name="h2o", version="1.3"),
    )
    parents = find_all_parents(frameworks.gama, frameworks)
    assert parents == []


@pytest.mark.parametrize("framework", ["gama", "h2o_automl"])
def test_find_all_parents_one_parent(framework):
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        gama_old=Namespace(name="gama_20.1.0", version="20.1.0", extends="gama"),
        h2o_automl=Namespace(name="h2o", version="latest"),
        h2o_automl_old=Namespace(name="h2o_1.2", version="1.2", extends="h2o_automl"),
    )
    parents = find_all_parents(frameworks[f"{framework}_old"], frameworks)
    assert parents == [frameworks[framework]]


@pytest.mark.parametrize("framework", ["gama", "h2o_automl"])
def test_find_all_parents_two_parents(framework):
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


def test_update_frameworks_with_parent_definitions_adds_parent_yaml():
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        gama_old=Namespace(name="gama_20.1.0", version="20.1.0", extends="gama"),
    )
    update_frameworks_with_parent_definitions(frameworks)
    assert frameworks.gama_old.description == "flexible automl"


def test_update_frameworks_with_parent_definitions_parent_overwrites_grandparent_yaml():
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        gama_old=Namespace(name="gama_2", version="1", description="automl", extends="gama"),
        gama_oldest=Namespace(name="gama_1", version="2", extends="gama_old"),
    )
    update_frameworks_with_parent_definitions(frameworks)
    assert frameworks.gama_oldest.description == "automl"


def test_update_frameworks_with_parent_definitions_does_not_overwrite_child_yaml():
    frameworks = Namespace(
        gama=Namespace(name="gama", version="latest", description="flexible automl"),
        gama_old=Namespace(name="gama_20.1", version="20.1", description="old gama", extends="gama"),
    )
    update_frameworks_with_parent_definitions(frameworks)
    assert frameworks.gama_old.description == "old gama"


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


def test_sanitize_and_add_defaults_module_not_overwritten(simple_resource):
    frameworks = Namespace(auto_sklearn=Namespace(module="custom_module"))
    sanitize_and_add_defaults(frameworks, simple_resource)
    assert frameworks.auto_sklearn.module == "custom_module"
