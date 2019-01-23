# TODO

### Global
1. README.md
2. HOWTO.md
3. more pydoc
4. unit tests

### Features
1. meta-benchmark? benchmark a subset of configured frameworks:\
`runbenchmark.py frameworks_list.yaml test`.
2. support list of sub-tasks in a benchmark:\
`runbenchmark.py {framework} -t task1 task2`.
3. visualizations for results.csv: can provide script generating simple plots using `matplotlib`.
4. AWS mode "recovery": could save locally all jobs ids that have been started in a given benchmark, so that if losing connection, a script could still automatically fetch output files from s3 to download and merge them.
4. progress bar?? fancy useless stuff.

### Improvements
1. AWS: reuse instances for faster startup, at least during a single benchmark, we could limit #instances = #parallel jobs.
2. timeouts (already in place for AWS, but could be implemented for each job individually).
3. search for `TODO` and `FIXME` in codebase.
