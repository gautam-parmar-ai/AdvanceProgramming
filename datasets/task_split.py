import torch
from torch.utils.data import Subset

from configs.config import Config


class TaskSplitter:

    def __init__(self, train_dataset, test_dataset):

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.num_tasks = Config.num_tasks
        self.classes_per_task = Config.classes_per_task

        self.task_class_map = self._create_task_map()

    def _create_task_map(self):

        task_map = {}

        for task_id in range(self.num_tasks):

            start_class = task_id * self.classes_per_task
            end_class = start_class + self.classes_per_task

            task_map[task_id] = list(range(start_class, end_class))

        return task_map

    def _filter_by_classes(self, dataset, class_list):

        indices = [
            i for i, (_, label) in enumerate(dataset)
            if label in class_list
        ]

        return Subset(dataset, indices)

    def get_task(self, task_id):

        class_list = self.task_class_map[task_id]

        train_subset = self._filter_by_classes(self.train_dataset, class_list)
        test_subset = self._filter_by_classes(self.test_dataset, class_list)

        return train_subset, test_subset

    def get_all_tasks(self):

        tasks = []

        for task_id in range(self.num_tasks):

            train_t, test_t = self.get_task(task_id)

            tasks.append({
                "task_id": task_id,
                "classes": self.task_class_map[task_id],
                "train": train_t,
                "test": test_t
            })

        return tasks
