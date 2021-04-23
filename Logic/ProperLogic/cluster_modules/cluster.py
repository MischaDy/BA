from Logic.ProperLogic.helper_classes.reducer import MaxReducer
from Logic.ProperLogic.misc_helpers import log_error
import torch

from itertools import count

import logging
logging.basicConfig(level=logging.INFO)


class Cluster:
    metric = 2

    def __init__(self, cluster_id, embeddings=None, embeddings_ids=None, label=None, center_point=None):
        """
        embeddings must be (flat) iterable of embeddings with len applicable
        :param embeddings:
        :param embeddings_ids:
        """
        if label is None:
            label = 'Unknown Person'
        self.label = label
        self.max_id_reducer = MaxReducer()
        if embeddings is None:
            self.embeddings_dict = dict()
            self.num_embeddings = 0
            self.center_point = None
            self.max_embedding_id = 0
            self.max_id_reducer(self.max_embedding_id)
        else:
            if embeddings_ids is None:
                embeddings_ids = count(1)
            # cast embeddings to dict
            self.embeddings_dict = dict(zip(embeddings_ids, embeddings))
            self.num_embeddings = len(self.embeddings_dict)
            if center_point is not None:
                self.center_point = center_point
            else:
                self.center_point = self.sum_embeddings(embeddings) / self.num_embeddings
            self.max_id_reducer.process_iterable(self.embeddings_dict.keys())
            self.max_embedding_id = self.max_id_reducer.get_state()

        self.cluster_id = cluster_id

    def set_label(self, label):
        self.label = label

    def set_cluster_id(self, cluster_id):
        self.cluster_id = cluster_id

    @classmethod
    def set_metric(cls, metric):
        cls.metric = metric

    def get_embeddings(self, with_embeddings_ids=False, as_dict=False, as_list=False):
        if with_embeddings_ids or as_dict:
            if as_dict:
                return self.embeddings_dict
            return self.embeddings_dict.items()

        embeddings = self.embeddings_dict.values()
        if as_list:
            return list(embeddings)
        return embeddings

    def get_embeddings_ids(self):
        return self.embeddings_dict.keys()

    def get_size(self):
        return len(self.embeddings_dict)

    def add_embedding(self, new_embedding, new_embedding_id=None, overwrite=False):
        return self.add_embeddings([new_embedding], [new_embedding_id], overwrite)

    def add_embeddings(self, new_embeddings, new_embeddings_ids=None, overwrite=False):
        if not new_embeddings:
            return

        if new_embeddings_ids is None:
            next_embedding_id = self.max_embedding_id + 1
            new_embeddings_ids = count(start=next_embedding_id)

        new_embeddings_dict = dict(zip(new_embeddings_ids, new_embeddings))
        if overwrite:
            self.embeddings_dict.update(new_embeddings_dict)
        else:
            new_embeddings_dict.update(self.embeddings_dict)
            self.embeddings_dict = new_embeddings_dict

        old_num_embeddings = self.num_embeddings
        self.num_embeddings = len(self.embeddings_dict)
        embeddings = self.get_embeddings(as_list=True)
        embeddings_sum = self.sum_embeddings(embeddings)

        if self.center_point is not None:
            self.center_point = (old_num_embeddings * self.center_point + embeddings_sum) / self.num_embeddings
        else:
            self.center_point = embeddings_sum / self.num_embeddings

    def remove_embedding_by_id(self, embedding_id):
        try:
            embedding = self.embeddings_dict.pop(embedding_id)
        except KeyError:
            log_error(f'embedding with id {embedding_id} not found.')
            return

        old_num_embeddings = self.num_embeddings
        self.num_embeddings -= 1
        # (old_center is a uniformly weighted sum of the old embeddings)
        try:
            self.center_point = (old_num_embeddings * self.center_point - embedding) / self.num_embeddings
        except ZeroDivisionError:  # num_embeddings is 0
            self.center_point = None

    def get_center_point(self):
        return self.center_point

    def get_embedding(self, embedding_id):
        return self.embeddings_dict[embedding_id]

    def contains_embedding(self, embedding_id):
        return self.embeddings_dict.get(embedding_id) is not None

    def compute_dist_to_center(self, embedding):
        return self.compute_dist(self.center_point, embedding)

    @classmethod
    def compute_dist(cls, embedding1, embedding2, metric=None):
        if metric is None:
            metric = cls.metric
        return float(torch.dist(embedding1, embedding2, p=metric))

    @staticmethod
    def sum_embeddings(embeddings):
        # return reduce(torch.add, embeddings)
        return torch.sum(torch.stack(embeddings), dim=0)
