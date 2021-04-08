import pytest

from amlb.frameworks import default_tag
from amlb.resources import Resources
from amlb.utils import Namespace as ns


@pytest.mark.parametrize(
    "frameworks, lookup, expected",
    [
        (ns(MixedCase=ns(name="MixedCase")), "MixedCase", "MixedCase"),
        (ns(MixedCase=ns(name="MixedCase")), "mixedcase", "MixedCase"),
        (ns(MixedCase=ns(name="MixedCase")), "MIXEDCASE", "MixedCase"),
        (ns(MixedCase=ns(name="MixedCase")), "mIxEdCasE", "MixedCase"),
    ]
)
def test_framework_definition_lookup_is_case_insensitive(frameworks, lookup, expected):
    res = ns(_frameworks={default_tag: frameworks})
    # binding `framework_definition` method to our resource mock: use pytest-mock instead?
    res.framework_definition = Resources.framework_definition.__get__(res)
    assert res.framework_definition(lookup) == (frameworks[expected], frameworks[expected].name)


def test_framework_definition_raises_error_if_no_matching_framework():
    res = ns(
        config=ns(frameworks=ns(definition_file="none")),
        _frameworks={default_tag: ns(present=ns(name="present"))}
    )
    # binding `framework_definition` method to our resource mock: use pytest-mock instead?
    res.framework_definition = Resources.framework_definition.__get__(res)
    assert res.framework_definition("present")
    with pytest.raises(ValueError, match=r"Incorrect framework `missing`"):
        res.framework_definition("missing")


def test_framework_definition_raises_error_if_the_framework_is_abstract():
    res = ns(
        config=ns(frameworks=ns(definition_file="none")),
        _frameworks={default_tag: ns(present=ns(name="present", abstract=True))}
    )
    # binding `framework_definition` method to our resource mock: use pytest-mock instead?
    res.framework_definition = Resources.framework_definition.__get__(res)
    with pytest.raises(ValueError, match=r"Framework definition `present` is abstract and cannot be run directly"):
        res.framework_definition("present")
