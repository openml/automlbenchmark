# AutoML Benchmarking

To run a benchmark call the `benchmark.py` file with three arguments.

1. The AutoML framework that should be evaluated, see [frameworks.json](resources/frameworks.json) for supported frameworks. If you want to add a framework see [here](docker/readme.md).
2. The benchmark suite to run. Should be one implemented in [benchmarks.json](resources/benchmarks.json)
3. If the benchmark should be run `local` or on `aws`.
4. (Optional) a file to apend the results to

## Run the benchmark local

A minimal example would be to run the test benchmarks with a random forest:

```
./benchmark.py RandomForest test local

\\some building output here

  benchmark_id  fold    result
0       test_1     0  0.933333
1       test_1     1  1.000000
2       test_2     0  0.811321
3       test_2     1  0.849057


```
## Run the benchmark on AWS

To run a benchmark on AWS you additionally need to

- Have `boto3` set up on you machine with access to your AWS account
- Change the name of the `ssh_key` and `sec_group` to values associated with you AWS account.

```
./benchmark.py RandomForest test aws

\\some building output here

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

  benchmark_id  fold              result
0       test_1     0  0.9333333333333333
1       test_1     1                 1.0
2       test_2     0  0.8679245283018868
3       test_2     1  0.8490566037735849

```
