from Logic.ProperLogic.misc_helpers import log_error
import torch

from functools import reduce
from itertools import count

import logging
logging.basicConfig(level=logging.INFO)


class Cluster:
    max_cluster_id = 1
    max_embedding_id = 1

    def __init__(self, embeddings=None, embeddings_ids=None, cluster_id=None, label=None, center_point=None):
        """
        embeddings must be (flat) iterable of embeddings with len applicable
        :param embeddings:
        :param embeddings_ids:
        """
        if label is None:
            label = 'Unknown Person'
        self.label = label
        if embeddings is None:
            self.embeddings = {}
            self.num_embeddings = 0
            self.center_point = None
            # max number ever assigned as id for an embedding in this cluster
            Cluster.max_embedding_id = 0
        else:
            # TODO: refactor
            # TODO: consistent type for embedding ids(?)
            if embeddings_ids is None:
                embeddings_ids = count(1)
            # cast embeddings to dict
            self.embeddings = dict(zip(embeddings_ids, embeddings))
            self.num_embeddings = len(self.embeddings)
            if center_point is not None:
                self.center_point = center_point
            else:
                self.center_point = Cluster.sum_embeddings(self.embeddings.values()) / self.num_embeddings
            Cluster.max_embedding_id = max(self.embeddings.keys())

        if cluster_id is not None:
            self.cluster_id = cluster_id
        else:
            self.cluster_id = Cluster.max_cluster_id
        Cluster.max_cluster_id = max(self.cluster_id, Cluster.max_cluster_id - 1) + 1

    def set_label(self, label):
        self.label = label

    def get_embeddings(self, with_embedding_ids=False):
        if with_embedding_ids:
            return self.embeddings.items()
        return self.embeddings.values()

    def get_size(self):
        return len(self.embeddings)

    def add_embedding(self, embedding, embedding_id=None, overwrite=False):
        if embedding_id is None:
            Cluster.max_embedding_id += 1
            embedding_id = Cluster.max_embedding_id
        if self.embeddings.get(embedding_id) is not None and not overwrite:
            raise RuntimeError('embedding with given ID already exists in this cluster')
        self.embeddings[embedding_id] = embedding

        old_num_embeddings = self.num_embeddings
        self.num_embeddings += 1
        # (old_center is a uniformly weighted sum of the old embeddings)
        self.center_point = (old_num_embeddings * self.center_point + embedding) / self.num_embeddings

    # def remove_embedding(self, embedding_id):
    #     # TODO: Needed?
    #     try:
    #         self.embeddings.pop(embedding_id)
    #     except ValueError as error:
    #         log_error('Specified embedding not found in embeddings')
    #         raise error
    #     old_num_embeddings = self.num_embeddings
    #     self.num_embeddings -= 1
    #     # (old_center is a uniformly weighted sum of the old embeddings)
    #     self.center_point = (old_num_embeddings * self.center_point - embedding_id) / self.num_embeddings

    def get_center_point(self):
        return self.center_point

    def compute_dist_to_center(self, embedding):
        return Cluster.compute_dist(self.center_point, embedding)

    @staticmethod
    def compute_dist(embedding1, embedding2):
        return float(torch.dist(embedding1, embedding2))

    @staticmethod
    def sum_embeddings(embeddings):
        return reduce(torch.add, embeddings)


class Clusters(list):
    def get_cluster_by_id(self, cluster_id):
        cluster_list = self.get_clusters_by_ids([cluster_id])
        try:
            return next(cluster_list)
        except StopIteration:
            log_error(f"no cluster with an id in '{cluster_id}' found")
            return None

    def get_clusters_by_ids(self, cluster_ids):
        # TODO: Improve efficiency! (Make dict with id as key?)
        return filter(lambda c: c.cluster_id in cluster_ids, self)
