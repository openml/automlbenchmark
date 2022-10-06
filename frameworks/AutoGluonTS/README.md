# AutoGluonTS

AutoGluonTS stands for autogluon.timeseries. This framework handles time series problems.

This code is currently a prototype, since time series support is not fully defined in AutoMLBenchmark yet.
Consider the code a proof of concept.

## Run Steps

To run AutoGluonTS in AutoMLBenchmark on the covid dataset from the AutoGluon tutorial, do the following:

1. Create a fresh Python environment
2. Follow automlbenchmark install instructions
3. Run the following command in terminal: ```python3 ../automlbenchmark/runbenchmark.py autogluonts ts test```

To run mainline AutoGluonTS instead of v0.5.2: ```python3 ../automlbenchmark/runbenchmark.py autogluonts:latest ts test```
