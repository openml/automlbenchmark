# TODO

### Global
1. README.md
2. HOWTO.md
2. setup.py to simplify installation
3. more pydoc
4. unit tests

### Features
1. support regression tasks: done for autosklearn, H2OAutoML, hyperoptsklearn, TPOT, RandomForest, others don't support it.
2. meta-benchmark? benchmark a subset of configured frameworks:\
`runbenchmark.py frameworks_list.yaml test`.
3. visualizations for results.csv: can provide script generating simple plots using `matplotlib`.
4. AWS mode "recovery": could save locally all jobs ids that have been started in a given benchmark, so that if losing connection, a script could still automatically fetch output files from s3 to download and merge them.
4. progress bar?? fancy useless stuff.

### Bugs & Improvements
1. properly kill job threads on KeyInterruptError (mainly in AWS mode)
1. review AWS termination logic: when is it stopped, when terminated
1. Fix input file transfer to EC2 instances: current logic can cause issues when running multiple benchmarks at the same time if using the same S3 bucket.
1. AWS: reuse instances for faster startup, at least during a single benchmark, we could limit #instances = #parallel jobs.
2. timeouts (already in place for AWS, but could be implemented for each job individually).
3. search for `TODO` and `FIXME` in codebase.
