import sys
sys.path.append("D:\\repositories/openml-python")

import openml
import yaml

benchmarks_to_include = [
    'resources/benchmarks/medium-8c4h.yaml',
    'resources/benchmarks/small-8c4h.yaml',
    'resources/benchmarks/large-8c4h.yaml'
]

if __name__ == '__main__':
    benchmarks = []
    for filename in benchmarks_to_include:
        with open(filename, 'r') as fh:
            benchmark = fh.read()
            benchmarks.append(yaml.load(benchmark))

    all_task_ids = []
    for problem in [problem for benchmark in benchmarks for problem in benchmark]:
        task_id = problem.get('openml_task_id')
        if task_id is not None:
            all_task_ids.append(task_id)

    suite = openml.study.create_benchmark_suite(
        name='AutoML Benchmark',
        description="Tasks of the ongoing AutoML benchmark, see https://openml.github.io/automlbenchmark/."
                    "The benchmark includes both binary and multiclass classification tasks.",
        task_ids=all_task_ids,
        alias="AutoML-Benchmark"
    )


