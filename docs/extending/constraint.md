# Constraints

Constraint definitions allow a set of common constraints to be applied to all tasks in 
a benchmark. Default constraint definitions are available in 
[`resources/constraint.yaml`](GITHUB/resources/constraint.yaml).
When no constraint is specified at the command line, the `test` constraint definition is used by default.

A constraint definition can consist of the following constraints:

- `folds` (default=10): The number of folds to evaluate for the task. Has to be less or equal to the number of folds defined by the task.
- `max_runtime_seconds` (default=3600): maximum time in seconds for each individual fold of a benchmark task. 
  This parameter is usually passed directly to the framework. If it doesn't respect the 
  constraint, the application will abort the task after `2 * max_runtime_seconds`. 
  In any case, the _actual_ time used is always recorded and available in the results.
- `cores` (default=-1): amount of cores used for each automl task. If non-positive, it will try to use all cores.
- `max_mem_size_mb` (default=-1): amount of memory assigned to each automl task. 
   If non-positive, then the amount of memory is computed from os available memory.
- `min_vol_size_mb` (default=-1): minimum amount of free space required on the volume. If non-positive, skips verification. If the requirement is not fulfilled, a warning message will be printed, but the task will still be attempted.
- `ec2_volume_type`: The volume type to use for the task when using EC2 instances, otherwise defaults to the value of `aws.ec2.volume_type` in your configuration file.

!!! warning "Constraints are not enforced!"

    These constraints are forwarded to the AutoML framework if possible but are 
    generally not enforced. Not all AutoML frameworks allow for e.g., memory limits
    to be set, and not all implementations that do treat it as a hard constraint.
    For that reason, only `max_runtime_seconds` is enforced as described above.
    It is advised when benchmarking to use an environment that mimics the given constraints.

??? info "Constraints can be overriden by `benchmark`"

    A benchmark definition can override constraints on a task level.
    This is useful if you want to define a benchmark which has different constraints
    for different tasks. The default "test" benchmark does this to limit runtime to
    60 seconds instead of 600 seconds, which is useful to get quick results for its
    small datasets. For more information, see [defining a benchmark](#ADD-link-to-adding-benchmark).


When writing your own constraint definition, it needs to be discoverable by the benchmark.
A good place to do this would be adding a `constraints.yaml` file to your benchmark
configuration directory (`~/.config/automlbenchmark` by default) and updating your
[custom configuration](../../using/configuration/#custom-configurations) by adding:

```yaml
benchmarks:                     
  constraints_file: 
    - '{root}/resources/constraints.yaml'
    - '{user}/constraints.yaml'
```

You can then define multiple constraints in your constraint file, for example:
```yaml title="{user}/constraints.yaml"
---

test:
  folds: 1
  max_runtime_seconds: 120

8h16c:
  folds: 10
  max_runtime_seconds: 28800
  cores: 16
  min_vol_size_mb: 65536
  ec2_volume_type: gp3
```

The new constraints can now be passed on the command line when executing the benchmark:
```bash
python runbenchmark.py randomforest validation 8h16c
```
*Note*: The above example is _allowed_ to run for 8 hours, but will stop earlier as 
`RandomForest` stops early after training 2000 trees.