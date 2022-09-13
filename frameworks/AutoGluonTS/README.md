# AutoGluonTS

AutoGluonTS stands for autogluon.timeseries. This framework handles time series problems.

This code is currently a prototype, since time series support is not fully defined in AutoMLBenchmark yet.
Consider the code a proof of concept.

## Run Steps

To run AutoGluonTS in AutoMLBenchmark on the covid dataset from the AutoGluon tutorial, do the following:

1. Create a fresh Python environment
2. Follow automlbenchmark install instructions
3. Run the following command in terminal: ```python3 ../automlbenchmark/runbenchmark.py autogluonts ts test```
4. Done.

To run mainline AutoGluonTS instead of v0.5.2: ```python3 ../automlbenchmark/runbenchmark.py autogluonts:latest ts test```

## TODO

### FIXME: Why does leaderboard claim a different test score than AutoMLBenchmark for RMSE?
### FIXME: Currently ignoring test_path, just using train data for evaluation
### TODO: How to evaluate more complex metrics like MAPE?
### How to pass timestamp_column?
### How to pass id_column?
### How to pass prediction_length?










