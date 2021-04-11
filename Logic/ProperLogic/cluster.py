import operator

from misc_helpers import log_error
import torch

from functools import reduce
from itertools import count

import logging
logging.basicConfig(level=logging.INFO)


class Cluster:
    def __init__(self, cluster_id, embeddings=None, embeddings_ids=None, label=None, center_point=None):
        """
        embeddings must be (flat) iterable of embeddings with len applicable
        :param embeddings:
        :param embeddings_ids:
        """
        if label is None:
            label = 'Unknown Person'
        self.label = label
        if embeddings is None:
            self.embeddings = dict()
            self.num_embeddings = 0
            self.center_point = None
            self.max_embedding_id = 0
        else:
            if embeddings_ids is None:
                embeddings_ids = count(1)
            # cast embeddings to dict
            self.embeddings = dict(zip(embeddings_ids, embeddings))
            self.num_embeddings = len(self.embeddings)
            if center_point is not None:
                self.center_point = center_point
            else:
                self.center_point = Cluster.sum_embeddings(self.embeddings.values()) / self.num_embeddings
            self.max_embedding_id = max(self.embeddings.keys())

        self.cluster_id = cluster_id

    def set_label(self, label):
        self.label = label

    def set_cluster_id(self, cluster_id):
        self.cluster_id = cluster_id

    def get_embeddings(self, with_embeddings_ids=False, as_dict=False):
        if with_embeddings_ids or as_dict:
            if as_dict:
                return self.embeddings
            return self.embeddings.items()
        return self.embeddings.values()

    def get_embeddings_ids(self):
        return self.embeddings.keys()

    def get_size(self):
        return len(self.embeddings)

    def add_embedding(self, embedding, embedding_id=None, overwrite=False):
        if embedding_id is None:
            self.max_embedding_id += 1
            embedding_id = self.max_embedding_id
        if self.embeddings.get(embedding_id) is not None and not overwrite:
            raise RuntimeError('embedding with given ID already exists in this cluster')
        self.embeddings[embedding_id] = embedding

        old_num_embeddings = self.num_embeddings
        self.num_embeddings += 1
        # (old_center is a uniformly weighted sum of the old embeddings)
        try:
            self.center_point = (old_num_embeddings * self.center_point + embedding) / self.num_embeddings
        except TypeError:  # center_point is None
            # TODO: Copy embedding or sth. like that instead of direct assignment?
            self.center_point = embedding

    def remove_embedding_by_id(self, embedding_id):
        try:
            embedding = self.embeddings.pop(embedding_id)
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
        return self.embeddings[embedding_id]

    def contains_embedding(self, embedding_id):
        return self.embeddings.get(embedding_id) is not None

    def compute_dist_to_center(self, embedding):
        return Cluster.compute_dist(self.center_point, embedding)

    @staticmethod
    def compute_dist(embedding1, embedding2):
        return float(torch.dist(embedding1, embedding2))

    @staticmethod
    def sum_embeddings(embeddings):
        return reduce(torch.add, embeddings)


class ClusterDict(dict):
    # TODO: Make sure constructor is only called when needed / doesn't produce more work than necessary!

    def __init__(self, clusters=None):
        super().__init__()
        if clusters is None:
            return

        for cluster in clusters:
            self[cluster.cluster_id] = cluster

    def get_clusters(self):
        return self.values()

    def get_cluster_by_id(self, cluster_id):
        try:
            return self[cluster_id]
        except KeyError:
            log_error(f"no cluster with id '{cluster_id}' found")
            return None

    def get_clusters_by_ids(self, cluster_ids):
        return map(self.get_cluster_by_id, cluster_ids)

    def get_cluster_ids(self):
        return self.keys()

    def get_cluster_labels(self, with_ids=False, unique=True):
        """
        If with_ids is provided, unique is ignored.

        :param with_ids:
        :param unique:
        :return:
        """
        attrs = ['cluster_id'] if with_ids else []
        attrs.append('label')
        cluster_labels = self.get_cluster_attrs(*attrs)

        if unique and not with_ids:
            return list(set(cluster_labels))
        return list(cluster_labels)

    def get_cluster_attrs(self, *attrs):
        clusters = self.get_clusters()
        attrs_getter = operator.attrgetter(*attrs)
        return map(attrs_getter, clusters)

    def reset_ids(self, start_id=1):
        clusters = self.get_clusters()
        old_ids = []
        for new_cluster_id, cluster in enumerate(clusters, start=start_id):
            old_ids.append(cluster.cluster_id)
            cluster.set_cluster_id(new_cluster_id)

        max_id = start_id + len(clusters)
        new_ids = list(range(start_id, max_id))
        return old_ids, new_ids

    def set_ids(self, old_ids, new_ids):
        clusters = self.get_clusters()
        old_to_new_ids_dict = dict(zip(old_ids, new_ids))
        for cluster in clusters:
            new_id = old_to_new_ids_dict[cluster.cluster_id]
            cluster.set_cluster_id(new_id)

    def any_cluster_with_emb(self, emb):
        clusters = self.get_clusters()
        return any(filter(lambda cluster: cluster.contains_embedding(emb), clusters))

    def add_clusters(self, clusters):
        for cluster in clusters:
            self[cluster.cluster_id] = cluster

    def add_cluster(self, cluster):
        self.add_clusters([cluster])

    def remove_clusters(self, clusters):
        for cluster in clusters:
            self.pop(cluster.cluster_id)

    def remove_cluster(self, cluster):
        self.remove_clusters([cluster])


