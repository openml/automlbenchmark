import pytest
from amlb.frameworks.definitions import _add_default_module, _add_default_version, \
    _add_default_setup_args, _add_default_setup_script, _add_default_setup_cmd, \
    _add_default_params, _add_default_image
from amlb.utils import Namespace


def test_params_set_to_empty_dict_if_undefined():
    framework = Namespace()
    _add_default_params(framework)
    assert framework.params == dict()


def test_params_is_converted_to_dict_if_defined():
    framework = Namespace(params=Namespace(my_param="set"))
    _add_default_params(framework)
    assert isinstance(framework.params, dict)
    assert framework.params["my_param"] == "set"


@pytest.mark.parametrize("name", ["a", "b"])
def test_module_default_uses_framework_name(name):
    dummy_config = Namespace(frameworks=Namespace(root_module="frameworks"))
    framework = Namespace(name=name)
    _add_default_module(framework, dummy_config)
    assert framework.module == f"frameworks.{name}"


@pytest.mark.parametrize("root", ["a", "b"])
def test_module_default_uses_root_module(root):
    dummy_config = Namespace(frameworks=Namespace(root_module=root))
    framework = Namespace(name="d")
    _add_default_module(framework, dummy_config)
    assert framework.module == f"{root}.d"


def test_module_is_not_replaced_if_defined():
    dummy_config = Namespace(frameworks=Namespace(root_module="frameworks"))
    framework = Namespace(name="c", module="custom")
    _add_default_module(framework, config=dummy_config)
    assert framework.module == "custom"


def test_version_is_set_to_stable_if_undefined():
    framework = Namespace()
    _add_default_version(framework)
    assert "version" in framework
    assert framework.version == "stable"


def test_version_is_kept_if_defined():
    framework = Namespace(version="v1.0")
    _add_default_version(framework)
    assert "version" in framework
    assert framework.version == "v1.0"


def test_setup_args_set_to_version_if_undefined():
    f_my_version = Namespace(version="my_version")
    f_my_other_version = Namespace(version="my_other_version")

    _add_default_setup_args(f_my_version)
    _add_default_setup_args(f_my_other_version)

    assert f_my_version.setup_args == ["my_version"]
    assert f_my_other_version.setup_args == ["my_other_version"]


@pytest.mark.parametrize("version, repo", [("my_version", "my_repo"), ("my_other_version", "my_other_repo")])
def test_setup_args_also_includes_repo_if_repo_is_defined(version, repo):
    f_my_version = Namespace(version=version, repo=repo)
    _add_default_setup_args(f_my_version)
    assert f_my_version.setup_args == [version, repo]


def test_setup_args_kept_if_defined():
    f_with_extra = Namespace(setup_args="w_extra", version="my_version", repo="my_repo")
    _add_default_setup_args(f_with_extra)
    assert f_with_extra.setup_args == ["w_extra"]


def test_setup_script_set_to_none_if_undefined(simple_resource):
    framework = Namespace()
    _add_default_setup_script(framework, simple_resource.config)
    assert framework.setup_script is None


def test_setup_script_kept_if_defined(simple_resource):
    framework = Namespace(module="my_module", setup_script="t.sh")
    _add_default_setup_script(framework, simple_resource.config)
    assert framework.setup_script == "t.sh"


def test_setup_script_interpolates_module(simple_resource):
    framework = Namespace(module="my_module", setup_script="{module}/t.sh")
    _add_default_setup_script(framework, simple_resource.config)
    assert framework.setup_script == "my_module/t.sh"


@pytest.mark.parametrize(
    "alias, actual",
    [
     ("input", "my_input"),
     ("output", "my_output"),
     ("user", "my_user_dir"),
     ("root", "my_root_dir"),
    ]
)
def test_setup_script_interpolates_directory(simple_resource, alias, actual):
    framework = Namespace(setup_script=f"{{{alias}}}/t.sh", module="")
    _add_default_setup_script(framework, simple_resource.config)
    assert framework.setup_script.endswith(f"{actual}/t.sh")


def test_setup_cmd_set_to_none_if_undefined(simple_resource):
    framework = Namespace()
    _add_default_setup_cmd(framework, simple_resource.config)
    assert framework.setup_cmd == None
    assert framework._setup_cmd == None


@pytest.mark.parametrize("commands", ["original", ["one", "two"]])
def test_setup_cmd_saves_original_if_defined(simple_resource, commands):
    framework = Namespace(setup_cmd=commands)
    _add_default_setup_cmd(framework, simple_resource.config)
    assert framework._setup_cmd == commands


def test_setup_cmd_converts_str_definition_to_list(simple_resource):
    framework = Namespace(setup_cmd="str_command")
    _add_default_setup_cmd(framework, simple_resource.config)
    assert framework.setup_cmd == ["str_command"]


def test_setup_cmd_does_not_convert_list_definition(simple_resource):
    framework = Namespace(setup_cmd=["str", "commands"])
    _add_default_setup_cmd(framework, simple_resource.config)
    assert framework.setup_cmd == ["str", "commands"]


@pytest.mark.parametrize(
    "alias, actual",
    [
     ("input", "my_input"),
     ("output", "my_output"),
     ("user", "my_user_dir"),
     ("root", "my_root_dir"),
    ]
)
def test_setup_cmd_interpolates_directory(simple_resource, alias, actual):
    framework = Namespace(setup_cmd=[f"{{{alias}}}"])
    _add_default_setup_cmd(framework, simple_resource.config)
    assert framework.setup_cmd[0].endswith(actual)


def test_image_uses_default_author(simple_resource):
    simple_resource.config.docker.image_defaults.author = "unit test author"
    framework = Namespace(version="v", name="n")
    _add_default_image(framework, simple_resource.config)
    assert framework.image.author == "unit test author"


def test_image_what_to_do_if_no_default_author(simple_resource):
    simple_resource.config.docker.image_defaults.author = None
    framework = Namespace(version="v", name="n")
    _add_default_image(framework, simple_resource.config)
    assert framework.image.author == ""


def test_image_author_kept_if_defined(simple_resource):
    simple_resource.config.docker.image_defaults.author = "author"
    framework = Namespace(name="n", version="0.1", image=Namespace(author="hfinley"))
    _add_default_image(framework, simple_resource.config)
    assert framework.image.author == "hfinley"


def test_image_uses_default_tag(simple_resource):
    simple_resource.config.docker.image_defaults.tag = "V1"
    framework = Namespace(version="V0.2dev", name="n")
    _add_default_image(framework, simple_resource.config)
    assert framework.image.tag == "V1"


def test_image_uses_lower_case_version_if_no_default_tag(simple_resource):
    simple_resource.config.docker.image_defaults.tag = None
    framework = Namespace(version="V0.2dev", name="n")
    _add_default_image(framework, simple_resource.config)
    assert framework.image.tag == "v0.2dev"


def test_image_tag_kept_if_defined(simple_resource):
    simple_resource.config.docker.image_defaults.tag = "v1.0dev"
    framework = Namespace(name="n", image=Namespace(tag="1.0-xenial"))
    _add_default_image(framework, simple_resource.config)
    assert framework.image.tag == "1.0-xenial"


def test_image_uses_default_image(simple_resource):
    simple_resource.config.docker.image_defaults.image = "img"
    framework = Namespace(name="decision_tree", version="v0.1")
    _add_default_image(framework, simple_resource.config)
    assert framework.image.image == "img"


def test_image_uses_lowercase_framework_name_if_no_default_image(simple_resource):
    simple_resource.config.docker.image_defaults.image = None
    framework = Namespace(name="Decision_tree", version="v0.1")
    _add_default_image(framework, simple_resource.config)
    assert framework.image.image == "decision_tree"


def test_image_image_kept_if_defined(simple_resource):
    simple_resource.config.docker.image_defaults.image = "default_image"
    framework = Namespace(name="n", version="20.0a", image=Namespace(image="automl"))
    _add_default_image(framework, simple_resource.config)
    assert framework.image.image == "automl"
