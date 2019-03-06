# TODO

### Global
1. README.md
1. HOWTO.md
1. setup.py to simplify installation
1. more pydoc
1. unit tests

### Features
1. meta-benchmark? benchmark a subset of configured frameworks:
    `runbenchmark.py frameworks_list.yaml test`.
1. progress bar?? fancy useless stuff.

### Bugs & Improvements
1. expose training_duration + count_models from ranger
1. properly kill job threads on KeyInterruptError (mainly in AWS mode)
1. AWS: reuse instances for faster startup, at least during a single benchmark, we could limit #instances = #parallel jobs.
1. search for `TODO` and `FIXME` in codebase.
