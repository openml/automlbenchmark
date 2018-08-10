# AutoML Benchmarking

To run a benchmark call the `benchmark.py` file with three arguments.

1. The AutoML framework that should be evaluated, see [frameworks.json](resources/frameworks.json) for supported frameworks. If you want to add a framework see [here](docker/readme.md).
2. The benchmark suite to run. Should be one implemented in [benchmarks.json](resources/benchmarks.json)
3. If the benchmark should be run `local` or on `aws`.
4. (Optional) a file to apend the results to

## Run the benchmark local

A minimal example would be to run the test benchmarks with a random forest:

```
./benchmark.py randomForest test local

[{'fold': 0, 'id': 59, 'result': '0.9333333333333333'},
 {'fold': 1, 'id': 59, 'result': '1.0'},
 {'fold': 0, 'id': 3913, 'result': '0.8490566037735849'},
 {'fold': 1, 'id': 3913, 'result': '0.8490566037735849'}]

```
## Run the benchmark on AWS

To run a benchmark on AWS you additionally need to

- Have `boto3` set up on you machine with access to you AWS account
- Change the name of the `ssh_key` and `sec_group` to values associated with you AWS account.

```
./benchmark.py randomForest test aws

Created 4 jobs
Starting instances
Instance pending
Instance pending
Instance pending
Instance pending
Instance pending
0/4 jobs done
0/4 jobs done
0/4 jobs done
0/4 jobs done
0/4 jobs done
0/4 jobs done
0/4 jobs done
0/4 jobs done
0/4 jobs done
0/4 jobs done
1/4 jobs done
1/4 jobs done
1/4 jobs done
1/4 jobs done
1/4 jobs done
1/4 jobs done
2/4 jobs done
4/4 jobs done
All jobs done!
Terminating Instances:
Termination successful
Termination successful
Termination successful
Termination successful
Termination successful
Termination successful
Termination successful
Termination successful
[{'fold': 0, 'id': 59, 'result': '0.9333333333333333'},
 {'fold': 1, 'id': 59, 'result': '1.0'},
 {'fold': 0, 'id': 3913, 'result': '0.8490566037735849'},
 {'fold': 1, 'id': 3913, 'result': '0.8490566037735849'}]

```
