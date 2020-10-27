import itertools

from input_output_logic import *
import torch

from functools import reduce

import logging

CLASSIFICATION_THRESHOLD = 1
_NUM_EMBEDDINGS_TO_CLASSIFY = 20

CLUSTER_SAVE_PATH = 'stored_clusters'
EMBEDDINGS_PATH = 'stored_embeddings'


# TODO: Proper comments!
# TODO: Store thumbnail imgs WITH the tensors


class Cluster:
    # TODO: Keep track of the cluster_id beyond runtime of the program??
    max_cluster_id = 1

    # TODO: Idea - assign ids to embeddings of each cluster which are valid at least for lifetime of the cluster
    # TODO: Don't allow accidental overwriting of embeddings by using same id!

    # TODO: How to handle embeddings-generator??
    def __init__(self, embeddings=None, embeddings_ids=None):
        """
        embeddings must be (flat) iterable of tensors with len applicable
        :param embeddings:
        :param embeddings_ids:
        """
        numeric_ids_assigned = False
        if embeddings is None:
            self.embeddings = {}
            self.num_embeddings = 0
            self.center_point = None
        else:
            # TODO?: refactor / improve efficiency!
            # TODO(?): consistent type for embedding ids
            if embeddings_ids is None:
                embeddings_ids = itertools.count(1)
                numeric_ids_assigned = True
            # cast embeddings to dict
            self.embeddings = dict(zip(embeddings_ids, embeddings))
            self.num_embeddings = len(self.embeddings)
            self.center_point = Cluster.sum_embeddings(self.embeddings.values()) / self.num_embeddings

        # max number ever assigned as id for an embedding in this cluster
        self.max_num_embedding_id = self.num_embeddings if numeric_ids_assigned else 0
        self.cluster_id = Cluster.max_cluster_id
        Cluster.max_cluster_id += 1

    def get_embeddings(self, return_embeddings_ids=False):
        if return_embeddings_ids:
            return self.embeddings
        return self.embeddings.values()

    def get_embedding_id(self, embedding):
        # TODO: implement(?)
        return ...

    def add_embedding(self, embedding, embedding_id=None):
        if embedding_id is None:
            self.max_num_embedding_id += 1
            embedding_id = self.max_num_embedding_id
        if self.embeddings.get(embedding_id):
            raise RuntimeError('embedding with given ID already exists in this cluster')
        self.embeddings[embedding_id] = embedding

        old_num_embeddings = self.num_embeddings
        self.num_embeddings += 1
        # (old_center is a uniformly weighted sum of the old embeddings)
        self.center_point = (old_num_embeddings * self.center_point + embedding) / self.num_embeddings

    def remove_embedding(self, embedding_id):
        # TODO: Better way to specify embedding? Assign ids or the like and output those when new embedding is added?
        try:
            self.embeddings.pop(embedding_id)
        except ValueError as error:
            log_error('Specified embedding not found in embeddings')
            raise error
        old_num_embeddings = self.num_embeddings
        self.num_embeddings -= 1
        # (old_center is a uniformly weighted sum of the old embeddings)
        self.center_point = (old_num_embeddings * self.center_point - embedding_id) / self.num_embeddings

    def get_center_point(self):
        return self.center_point

    def compute_dist_to_center(self, embedding):
        return float(torch.dist(self.center_point, embedding))

    def save_cluster(self, save_path):
        """

        :param save_path:
        :return:
        """
        cluster_save_path = os.path.join(save_path, f"cluster_{self.cluster_id}")
        save_cluster_embeddings_to_path(self.embeddings, cluster_save_path)

    @classmethod
    def load_cluster(cls, path_to_cluster, cluster_id=None):
        """

        :param path_to_cluster:
        :param cluster_id:
        :return:
        """
        embeddings = list(load_embeddings_from_path(path_to_cluster))
        cluster = cls(embeddings)
        if cluster_id is not None:
            cluster.cluster_id = cluster_id
        return cluster

    @staticmethod
    def sum_embeddings(embeddings):
        return reduce(torch.add, embeddings)

    # def compute_avg_dist_to_center(self, dim=None):
    #     pass


def main_algorithm(embeddings_path, classification_threshold, cluster_save_path=None):
    """
    Build clusters from face embeddings stored in the given path using the specified classification threshold.
    (Currently handled as: All embeddings closer than the distance given by the classification threshold are placed in
    the same cluster. If cluster_save_path is set, store the resulting clusters as directories in the given path.

    :param embeddings_path:
    :param classification_threshold:
    :param cluster_save_path:
    :return:
    """
    embeddings_loader = load_embeddings_from_path(embeddings_path, yield_paths=True)
    try:
        first_file_path, first_embedding = next(embeddings_loader)
    except StopIteration as error:
        log_error('No embeddings found in path')
        raise error
    first_file_name = os.path.split(first_file_path)[-1]
    clusters = [Cluster([first_embedding], [first_file_name])]
    # iterate over remaining embeddings
    # TODO: replace by enumerate iterator
    for counter, (embedding_file_path, embedding) in zip(range(2, _NUM_EMBEDDINGS_TO_CLASSIFY + 1), embeddings_loader):
        logging.info(f'Current embedding number: {counter}')

        shortest_dist, nearest_cluster = float('inf'), None
        # TODO: Use map and min etc. to find nearest cluster and shortest dist!
        for cluster in clusters:
            dist_to_center = cluster.compute_dist_to_center(embedding)
            if dist_to_center < shortest_dist:
                shortest_dist = dist_to_center
                nearest_cluster = cluster
        if shortest_dist <= classification_threshold:
            nearest_cluster.add_embedding(embedding, embedding_file_path)
        else:
            clusters.append(Cluster([embedding], [embedding_file_path]))

    if cluster_save_path is not None:
        for cluster in clusters:
            cluster.save_cluster(cluster_save_path)


def log_error(msg):
    logging.error('Error: ' + msg)


if __name__ == '__main__':
    main_algorithm(EMBEDDINGS_PATH, CLASSIFICATION_THRESHOLD, CLUSTER_SAVE_PATH)
