Installation:


## Add a benchmark

In this section, `benchmark` means a suite of datasets that can be used to feed any of the available frameworks, in combination with a set of constraints (time limit, cpus, memory) enforced by the application.

A benchmark definition will then consist in a [datasets definition](#datasets-definition) and a [constraints definition](#constraint-definition).

Each dataset must contain a training set and a test set. There can be multiple training/test splits, in which case each split is named a `fold`, so that the same dataset can be benchmarked multiple times using a different fold.


or using pyenv:
```bash
pyenv install {python_version: 3.9.16}
pyenv virtualenv ve-automl
pyenv local ve-automl
```

- _**NOTE**: in case of issues when installing Python requirements, you may want to try the following:_
    - _on some platforms, we need to ensure that requirements are installed sequentially:_ `xargs -L 1 python -m pip install < requirements.txt`.
    - _enforce the `python -m pip` version above in your virtualenv:_ `python -m pip install --upgrade pip==19.3.1`.




## Troubleshooting guide

### Where are the results?

By default, the results for a benchmark execution are made available in a subfolder under `output_dir` (if not specified by `-o my_results`, then this is under `{cwd}/results`).

This subfolder is named `{framework}_{benchmark}_{constraint}_{mode}_{timestamp}`.

So that for example:
```bash
python runbenchmark.py randomforest
```
will create a subfolder `randomforest_test_test_local_20200108T184305`,

and:
```bash
python runbenchmark.py randomforest validation 1h4c -m aws
```
will create a subfolder `randomforest_validation_1h4c_aws_20200108T184305`.


Then each subfolder contains:
 - a `score` folder with a `results.csv` file concatenating the results from all the tasks in the benchmark, as well as potentially other individual results for each task.  
 - a `predictions` folder with the predictions for each task in the benchmark.
 - a `logs` folder: only if benchmark was executed with `-o output_dir` argument.
 - possibly more folders if the framework saves additional artifacts.
 
 Also the `output_dir` contains a `results.csv` concatenating **ALL results** from all subfolders.


### Where are the logs?

By default the application logs are available under `{cwd}/logs` if the benchmark is executed without specifying the `output_dir`, otherwise, they'll be available under the `logs` subfolder in the benchmark results (see [Where are the results?](#where-are-the-results)).

The application can collect various logs:
- local benchmark application logs: those are always collected. For each run, the application generated 2 log files locally:
  - `runbenchmark_{timestamp}.log`: contains logs for the application only (from DEBUG level).
  - `runbenchmark_{timestamp}_full.log`: contains logs for the application + other Python libraries (from INFO level); e.g. `boto3` logs when running in `aws` mode.
- remote application logs: for `aws` mode only, logs generated on the remote instances are automatically downloaded to the results folder, together with other result artifacts. 
- framework logs (optional): if the framework integration supports it, it is possible to ask for the framework logs by creating a custom framework definition as follow:
  ```yaml
  H2OAutoML:
    extends: H2OAutoML
    params:
      _save_artifacts: ['logs']
  ```

### Profiling the application

Currently, the application provides a global flag `--profiling` to activate profiling for some specific methods that can be slow or memory intensive:
```bash
python runbenchmark.py randomforest --profiling
```
All methods/functions are not profiled, but if you need to profile more, you just need to decorate the function with the `@profile()` decorator (from `amlb.utils`).

#### Memory usage
_Examples of memory info when using this custom profiling_:
```text
[PROFILING] `amlb.datasets.openml.OpenmlDatasplit.data` returned object size: 45.756 MB.
[PROFILING] `amlb.datasets.openml.OpenmlDatasplit.data` memory change; process: +241.09 MB/379.51 MB, resident: +241.09 MB/418.00 MB, virtual: +230.01 MB/4918.16 MB.
...
[PROFILING] `amlb.data.Datasplit.release` executed in 0.007s.
[PROFILING] `amlb.data.Datasplit.release` memory change; process: -45.73 MB/238.80 MB, resident: +0.00 MB/414.60 MB, virtual: +0.00 MB/4914.25 MB.
```

#### Methods duration
_Examples of method duration info when using this custom profiling_:
```text
[PROFILING] `amlb.datasets.openml.OpenmlLoader.load` executed in 7.456s.
...
[PROFILING] `amlb.data.Datasplit.X_enc` executed in 6.570s.
```

### Python library version conflict 
see [Framework integration](#frameworks-requiring-a-dedicated-virtual-env)

### Framework setup is not executed
Try the following:
- force the setup using the `-s only` or `-s force` arg on the command line:
  - `-s only` or `--setup=only` will force the setup and skip the benchmark run.
  - `-s force` or `--setup=force` will force the setup and run the benchmark immediately.
- delete the `.marker_setup_safe_to_delete` from the framework module and try to run the benchmark again. This marker file is automatically created after a successful setup to avoid having to execute it each tine (setup phase can be time-consuming), this marker then prevents auto-setup, except if the `-s only` or `-s force` args above are used.

### Framework setup fails
If the setup fails, first note that only the following OS are fully supported:
- Ubuntu 18.04

The setup is created for Debian-based linux environments, and macOS (most frameworks can be installed on macOS, ideally with `brew` installed, but there may be a few exceptions), so it may work with other Linux environments not listed above (e.g. Debian, Ubuntu 20.04, ...).
The best way to run benchmarks on non-supported OS, is to use the docker mode.

If the setup fails on a supported environment, please try the following:
- force the setup: see above.
- ensure that the same framework is not set up multiple times in parallel on the same machine:
  - first use `python runbenchmark.py MyFramework -s only` on one terminal.
  - then you can trigger multiple `python runbenchmark.py MyFramework ...` (without `-s` option) in parallel.
- delete the `lib` and `venv` folders, if present, under the given framework folder (e.g. `frameworks/MyFramework`), and try the setup again.


[README]: ./README.md
[OpenML]: https://openml.org
[ARFF]: https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
[CSV]: https://tools.ietf.org/html/rfc4180
[Docker]: https://docs.docker.com/
[config]: ../resources/config.yaml
