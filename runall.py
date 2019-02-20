import argparse

parser = argparse.ArgumentParser()
parser.add_argument('framework', type=str,
                    help="The framework to evaluate as defined by default in resources/frameworks.yaml.")
parser.add_argument('benchmark', type=str, nargs='?', default='test',
                    help="The benchmark type to run as defined by default in resources/benchmarks/{benchmark}.yaml.")
parser.add_argument('-m', '--mode', choices=['local', 'docker', 'aws'], default='local',
                    help="The mode that specifies how/where the benchmark tasks will be running. Defaults to %(default)s.")
parser.add_argument('-p', '--parallel', metavar='parallel_jobs', type=int, default=1,
                    help="The number of jobs (i.e. tasks or folds) that can run in parallel. Defaults to %(default)s. "
                         "Currently supported only in docker and aws mode.")
args = parser.parse_args()

