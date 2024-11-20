from amlb.benchmarks.openml import is_openml_benchmark


def test_is_openml_benchmark_valid():
    assert is_openml_benchmark("openml/s/271")
    assert is_openml_benchmark("test.openml/s/271")
    assert is_openml_benchmark("openml/t/41")

def test_is_openml_benchmark_not_valid():
    assert not is_openml_benchmark("openml/d/271"), "datasets are not valid targets"
    assert not is_openml_benchmark("api.openml/t/271"), "only openml and test.openml are valid prefixes"
    assert not is_openml_benchmark("validation"), "this is a file-defined benchmark"
