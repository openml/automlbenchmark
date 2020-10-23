import pytest
from amlb.resources import Resources
from amlb.utils import Namespace as NS


@pytest.mark.parametrize(
    "frameworks, lookup, expected",
    [
        (NS(MixedCase=NS(name="MixedCase")), "MixedCase", "MixedCase"),
        (NS(MixedCase=NS(name="MixedCase")), "mixedcase", "MixedCase"),
        (NS(MixedCase=NS(name="MixedCase")), "MIXEDCASE", "MixedCase"),
        (NS(MixedCase=NS(name="MixedCase")), "mIxEdCasE", "MixedCase"),
    ]
)
def test_framework_definition_lookup_is_case_insensitive(frameworks, lookup, expected):
    res = NS(_frameworks=frameworks)
    # binding `framework_definition` method to our resource mock: use pytest-mock instead?
    res.framework_definition = Resources.framework_definition.__get__(res)
    assert res.framework_definition(lookup) == (frameworks[expected], frameworks[expected].name)


def test_framework_definition_raises_error_if_no_matching_framework():
    res = NS(
        config=NS(frameworks=NS(definition_file="none")),
        _frameworks=NS(present=NS(name="present"))
    )
    # binding `framework_definition` method to our resource mock: use pytest-mock instead?
    res.framework_definition = Resources.framework_definition.__get__(res)
    assert res.framework_definition("present")
    with pytest.raises(ValueError, match=r"Incorrect framework `missing`"):
        res.framework_definition("missing")
