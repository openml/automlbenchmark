# Parameters of `runbenchmark.py`

The parameters of the `runbenchmark.py` script can be shown with:

```{ .text title="python runbenchmark.py --help" .limit_max_height }
usage: runbenchmark.py [-h] [-m {local,aws,docker,singularity}] [-t [task_id ...]] [-f [fold_num ...]] [-i input_dir] [-o output_dir] [-u user_dir] [-p parallel_jobs] [-s {auto,skip,force,only}] [-k [true|false]]
                       [-e] [--logging LOGGING] [--openml-run-tag OPENML_RUN_TAG]
                       framework [benchmark] [constraint]

positional arguments:
  framework             The framework to evaluate as defined by default in resources/frameworks.yaml.
                        To use a labelled framework (i.e. a framework defined in resources/frameworks-{label}.yaml),
                        use the syntax {framework}:{label}.
  benchmark             The benchmark type to run as defined by default in resources/benchmarks/{benchmark}.yaml,
                        a path to a benchmark description file, or an openml suite or task.
                        OpenML references should be formatted as 'openml/s/X' and 'openml/t/Y',
                        for studies and tasks respectively. Use 'test.openml/s/X' for the 
                        OpenML test server.
                        (default: 'test')
  constraint            The constraint definition to use as defined by default in resources/constraints.yaml.
                        (default: 'test')

optional arguments:
  -h, --help            show this help message and exit
  -m {local,aws,docker,singularity}, --mode {local,aws,docker,singularity}
                        The mode that specifies how/where the benchmark tasks will be running.
                        (default: 'local')
  -t [task_id ...], --task [task_id ...]
                        The specific task name (as defined in the benchmark file) to run.
                        When an OpenML reference is used as benchmark, the dataset name should be used instead.
                        If not provided, then all tasks from the benchmark will be run.
  -f [fold_num ...], --fold [fold_num ...]
                        If task is provided, the specific fold(s) to run.
                        If fold is not provided, then all folds from the task definition will be run.
  -i input_dir, --indir input_dir
                        Folder from where the datasets are loaded by default.
                        (default: '/Users/pietergijsbers/.openml')
  -o output_dir, --outdir output_dir
                        Folder where all the outputs should be written.(default: '/Users/pietergijsbers/repositories/forks/automlbenchmark-fork/results')
  -u user_dir, --userdir user_dir
                        Folder where all the customizations are stored.(default: '~/.config/automlbenchmark')
  -p parallel_jobs, --parallel parallel_jobs
                        The number of jobs (i.e. tasks or folds) that can run in parallel.
                        A hard limit is defined by property `job_scheduler.max_parallel_jobs`
                         in `resources/config.yaml`.
                        Override this limit in your custom `config.yaml` file if needed.
                        Supported only in aws mode or container mode (docker, singularity).
                        (default: 1)
  -s {auto,skip,force,only}, --setup {auto,skip,force,only}
                        Framework/platform setup mode. Available values are:
                        • auto: setup is executed only if strictly necessary.
                        • skip: setup is skipped.
                        • force: setup is always executed before the benchmark.
                        • only: only setup is executed (no benchmark).
                        (default: 'auto')
  -k [true|false], --keep-scores [true|false]
                        Set to true (default) to save/add scores in output directory.
  -e, --exit-on-error   If set, terminates on the first task that does not complete with a model.
  --logging LOGGING     Set the log levels for the 3 available loggers:
                        • console
                        • app: for the log file including only logs from amlb (.log extension).
                        • root: for the log file including logs from libraries (.full.log extension).
                        Accepted values for each logger are: notset, debug, info, warning, error, fatal, critical.
                        Examples:
                          --logging=info (applies the same level to all loggers)
                          --logging=root:debug (keeps defaults for non-specified loggers)
                          --logging=console:warning,app:info
                        (default: 'console:info,app:debug,root:info')
  --openml-run-tag OPENML_RUN_TAG
                        Tag that will be saved in metadata and OpenML runs created during upload, must match '([a-zA-Z0-9_\-\.])+'.
```


## Profiling the application

Currently, the application provides a global flag `--profiling` to activate profiling 
for some specific methods that can be slow or memory intensive:

```bash
python runbenchmark.py randomforest --profiling
```

Not all methods and functions are not profiled, but if you need to profile more, 
you just need to decorate the function with the `@profile()` decorator (from `amlb.utils`).
Profiling reports on memory usage and function durations:

```{ .text title="Example of profiling logs" }
[PROFILING] `amlb.datasets.openml.OpenmlLoader.load` executed in 7.456s.
[PROFILING] `amlb.datasets.openml.OpenmlDatasplit.data` returned object size: 45.756 MB.
[PROFILING] `amlb.datasets.openml.OpenmlDatasplit.data` memory change; process: +241.09 MB/379.51 MB, resident: +241.09 MB/418.00 MB, virtual: +230.01 MB/4918.16 MB.
[PROFILING] `amlb.data.Datasplit.X_enc` executed in 6.570s.
[PROFILING] `amlb.data.Datasplit.release` executed in 0.007s.
[PROFILING] `amlb.data.Datasplit.release` memory change; process: -45.73 MB/238.80 MB, resident: +0.00 MB/414.60 MB, virtual: +0.00 MB/4914.25 MB.
```