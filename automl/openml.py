import openml


class OpenML(object):

    default_api_key = 'c1994bdb7ecb3c6f3c8f3b35f4b47f1f'

    def __init__(self, api_key=default_api_key):
        super().__init__()
        openml.config.key = api_key

    def load(self, task_id, fold):
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, Y = task.get_X_and_y()
        train, test = task.get_train_test_split_indices(fold)

