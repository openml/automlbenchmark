import arff
import requests
import yaml

# I don't have OpenML installed locally
import sys
sys.path.append("D:\\repositories/openml-python/")
import openml

# benchmark configurations
small_config_url = "https://raw.githubusercontent.com/openml/automlbenchmark/master/resources/benchmarks/small-8c4h.yaml"
medium_config_url = "https://raw.githubusercontent.com/openml/automlbenchmark/master/resources/benchmarks/medium-8c4h.yaml"

# auto-sklearn problems
binary_url = "https://raw.githubusercontent.com/automl/auto-sklearn/master/autosklearn/metalearning/files/roc_auc_binary.classification_dense/algorithm_runs.arff"
multiclass_url = "https://raw.githubusercontent.com/automl/auto-sklearn/master/autosklearn/metalearning/files/log_loss_multiclass.classification_dense/algorithm_runs.arff"

print('loading files')
small_configuration = yaml.load(requests.get(small_config_url).text)
medium_configuration = yaml.load(requests.get(medium_config_url).text)

binary_configuration = arff.loads(requests.get(binary_url).text)
multiclass_configuration = arff.loads(requests.get(multiclass_url).text)


print('parsing files')
benchmark_tids = set([problem.get('openml_task_id') for problem in small_configuration]
                     + [problem.get('openml_task_id') for problem in medium_configuration])

autosklearn_tids = set([int(row[0]) for row in binary_configuration['data']]
                       + [int(row[0]) for row in multiclass_configuration['data']])

print('comparing tids')
print(benchmark_tids & autosklearn_tids)


print('retrieving and comparing dids')


def try_get_did_for_task(tid):
    try:
        return openml.tasks.get_task(tid, download_data=False).dataset_id
    except:
        print('Failed to get task', tid)


benchmark_dids = set([try_get_did_for_task(tid) for tid in benchmark_tids if tid is not None])
autosklearn_dids = set((try_get_did_for_task(tid) for tid in autosklearn_tids if tid is not None))

print(set(benchmark_dids) & set(autosklearn_dids))
