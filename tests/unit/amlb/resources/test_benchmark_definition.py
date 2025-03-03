from functools import partial

import pytest

from amlb import Resources
from amlb.utils import Namespace


@pytest.fixture
def amlb_dummy_configuration() -> Namespace:
    defaults = {
        "max_runtime_seconds": 0,
        "cores": 1,
        "folds": 2,
        "max_mem_size_mb": 3,
        "min_vol_size_mb": 4,
        "quantile_levels": 5,
    }

    aws_defaults = {
        "ec2": {
            "volume_type": "gp3",
            "instance_type": {
                "series": "m5",
                "map": {"4": "small", "default": "large"},
            },
        }
    }
    return Namespace(
        aws=Namespace.from_dict(aws_defaults),
        benchmarks=Namespace(defaults=Namespace.from_dict(defaults)),
    )


def test_validate_task_strict_requires_name():
    with pytest.raises(ValueError) as excinfo:
        Resources._validate_task(
            task=Namespace(),
            config_=Namespace(),
            lenient=False,
        )
    assert "mandatory properties as missing" in excinfo.value.args[0]


def test_validate_task_strict_requires_id(amlb_dummy_configuration: Namespace):
    strict_validate = partial(
        Resources._validate_task, config_=amlb_dummy_configuration, lenient=False
    )
    with pytest.raises(ValueError) as excinfo:
        strict_validate(task=Namespace(name="foo"))
    assert "must contain an ID or one property" in excinfo.value.args[0]


@pytest.mark.parametrize(
    ("properties", "expected"),
    [
        (Namespace(id="bar"), "bar"),
        (Namespace(openml_task_id=42), "openml.org/t/42"),
        (Namespace(openml_dataset_id=42), "openml.org/d/42"),
        (Namespace(dataset="bar"), "bar"),
        (Namespace(dataset=Namespace(id="bar")), "bar"),
    ],
)
def test_validate_task_id_formatting(
    properties: Namespace, expected: str, amlb_dummy_configuration: Namespace
):
    task = Namespace(name="foo") | properties
    Resources._validate_task(task=task, config_=amlb_dummy_configuration)
    assert task["id"] == expected


def test_validate_task_adds_benchmark_defaults(amlb_dummy_configuration: Namespace):
    task = Namespace(name=None)
    Resources._validate_task(task, amlb_dummy_configuration, lenient=True)

    config = Namespace.dict(amlb_dummy_configuration, deep=True)
    for setting, default in config["benchmarks"]["defaults"].items():
        assert task[setting] == default
    assert task["ec2_volume_type"] == amlb_dummy_configuration.aws.ec2.volume_type


def test_validate_task_does_not_overwrite(amlb_dummy_configuration: Namespace):
    task = Namespace(name=None, cores=42)
    Resources._validate_task(task, amlb_dummy_configuration, lenient=True)

    config = Namespace.dict(amlb_dummy_configuration, deep=True)
    assert task.cores == 42
    for setting, default in config["benchmarks"]["defaults"].items():
        if setting != "cores":
            assert task[setting] == default


def test_validate_task_looks_up_instance_type(amlb_dummy_configuration: Namespace):
    instance_type = amlb_dummy_configuration.aws.ec2.instance_type
    reverse_size_map = {v: k for k, v in Namespace.dict(instance_type.map).items()}
    n_cores_for_small = int(reverse_size_map["small"])

    task = Namespace(name="foo", cores=n_cores_for_small)
    Resources._validate_task(task, amlb_dummy_configuration, lenient=True)
    assert task["ec2_instance_type"] == "m5.small", (
        "Should resolve to the instance type with the exact amount of cores"
    )

    task = Namespace(name="foo", cores=n_cores_for_small - 1)
    Resources._validate_task(task, amlb_dummy_configuration, lenient=True)
    assert task["ec2_instance_type"] == "m5.small", (
        "If exact amount of cores are not available, should resolve to next biggest"
    )

    task = Namespace(name="foo", cores=n_cores_for_small + 1)
    Resources._validate_task(task, amlb_dummy_configuration, lenient=True)
    assert task["ec2_instance_type"] == "m5.large", (
        "If bigger than largest in map, should revert to default"
    )

    task = Namespace(name="foo", ec2_instance_type="bar")
    Resources._validate_task(task, amlb_dummy_configuration, lenient=True)
    assert task["ec2_instance_type"] == "bar", (
        "Should not overwrite explicit configuration"
    )
