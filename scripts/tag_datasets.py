import sys
sys.path.append("D:\\repositories/openml-python")

import openml

if __name__ == '__main__':
    suite = openml.study.get_suite(218)
    for taskid in suite.tasks:
        print('collecting t/', taskid)
        task = openml.tasks.get_task(taskid, download_data=False)
        print('collecting d/', taskid)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
        print('tagging')
        dataset.push_tag('study_218')
