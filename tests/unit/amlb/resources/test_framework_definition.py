import pytest
from amlb.resources import Resources
from amlb.utils import Namespace as NS


def test_framework_definition_lookup_is_case_insensitive():
    res = NS(
        _frameworks=NS(
            lower=NS(name="lower"),
            UPPER=NS(name="UPPER"),
            Camel=NS(name="CaMel")
        )
    )
    # binding `framework_definition` method to our resource mock: use pytest-mock instead?
    res.framework_definition = Resources.framework_definition.__get__(res)

    assert res.framework_definition("lower") == (res._frameworks.lower, "lower")
    assert res.framework_definition("upper") == (res._frameworks.UPPER, "UPPER")
    assert res.framework_definition("camel") == (res._frameworks.Camel, "CaMel")
    assert res.framework_definition("CAMEL") == (res._frameworks.Camel, "CaMel")
    assert res.framework_definition("Camel") == (res._frameworks.Camel, "CaMel")


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
