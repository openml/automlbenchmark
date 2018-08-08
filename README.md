# AutoML Benchmarking

To run a benchmark call the `benchmark.py` file with three arguments.

1. The AutoML framework that should be evaluated, see [frameworks.json](resources/frameworks.json) for supported frameworks. If you want to add a framework see [here](docker/readme.md).
2. The benchmark suite to run. Should be one implemented in [benchmarks.json](resources/benchmarks.json)
3. If the benchmark should be run `local` or on `aws`.

A minimal example would be to locally run the test benchmarks with an random forest:

```
./benchmark.py randomForest test local
```

This returns a dictionary with the performance values on the test benchmarks.
