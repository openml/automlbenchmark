# TODO

### Global
1. README.md
1. HOWTO.md
1. setup.py to simplify installation
1. more pydoc
1. unit tests

### Features
1. AWS mode "recovery": could save locally all jobs ids that have been started in a given benchmark, so that if losing connection, a script could still automatically fetch output files from s3 to download and merge them.
1. meta-benchmark? benchmark a subset of configured frameworks:
    `runbenchmark.py frameworks_list.yaml test`.
1. visualizations for results.csv: can provide script generating simple plots using `matplotlib`.
1. progress bar?? fancy useless stuff.

### Bugs & Improvements
1. expose training_duration + count_models from ranger
1. properly kill job threads on KeyInterruptError (mainly in AWS mode)
1. AWS: reuse instances for faster startup, at least during a single benchmark, we could limit #instances = #parallel jobs.
1. timeouts (already in place for AWS, but could be implemented for each job individually).
1. search for `TODO` and `FIXME` in codebase.
