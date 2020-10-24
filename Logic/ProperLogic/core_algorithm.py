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

    # TODO: How to handle embeddings-generator??
    def __init__(self, embeddings=None):
        """
        embeddings can be single tensor or (flat) iterable of tensors
        :param embeddings:
        """
        if embeddings is None:
            self.embeddings = []
            self.num_embeddings = 0
            self.center_point = None
        else:
            # cast embeddings to list - brackets around it if it's a single tensor, otherwise directly cast to list
            self.embeddings = list(embeddings) if not isinstance(embeddings, torch.Tensor) else [embeddings]
            self.num_embeddings = len(self.embeddings)
            self.center_point = Cluster.sum_embeddings(embeddings) / self.num_embeddings
        self.cluster_id = Cluster.max_cluster_id
        Cluster.max_cluster_id += 1

    def add_embedding(self, embedding):
        self.embeddings.append(embedding)
        old_num_embeddings = self.num_embeddings
        self.num_embeddings += 1
        # (old_center is a uniformly weighted sum of the old embeddings)
        self.center_point = (old_num_embeddings * self.center_point + embedding) / self.num_embeddings

    # def add_embeddings(self, new_embeddings):
    #     self.embeddings.extend(new_embeddings)
    #     self.num_embeddings += len(new_embeddings)
    #     self.center_point = ...  # TODO!

    def remove_embedding(self, embedding):
        # TODO: Better way to specify embedding? Assign ids or the like and output those when new embedding is added?
        try:
            self.embeddings.remove(embedding)
        except ValueError as error:
            log_error('Specified embedding not found in embeddings')
            raise error
        old_num_embeddings = self.num_embeddings
        self.num_embeddings -= 1
        # (old_center is a uniformly weighted sum of the old embeddings)
        self.center_point = (old_num_embeddings * self.center_point - embedding) / self.num_embeddings

    # def remove_embeddings(self, embeddings_to_remove):
    #     # TODO!
    #     pass

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
    embeddings_loader = load_embeddings_from_path(embeddings_path, yield_paths=False)
    try:
        first_embedding = next(embeddings_loader)
    except StopIteration as error:
        log_error('No embeddings found in path')
        raise error
    clusters = [Cluster(first_embedding)]
    # iterate over remaining embeddings
    # TODO: replace by enumerate iterator
    for counter, embedding in zip(range(2, _NUM_EMBEDDINGS_TO_CLASSIFY+1), embeddings_loader):
        logging.info(f'Current embedding number: {counter}')

        shortest_dist, nearest_cluster = float('inf'), None
        # TODO: Use map and min etc. to find nearest cluster and shortest dist!
        for cluster in clusters:
            dist_to_center = cluster.compute_dist_to_center(embedding)
            if dist_to_center < shortest_dist:
                shortest_dist = dist_to_center
                nearest_cluster = cluster
        if shortest_dist <= classification_threshold:
            nearest_cluster.add_embedding(embedding)
        else:
            clusters.append(Cluster(embedding))

    if cluster_save_path is not None:
        for cluster in clusters:
            cluster.save_cluster(cluster_save_path)


def log_error(msg):
    logging.error('Error: ' + msg)


if __name__ == '__main__':
    main_algorithm(EMBEDDINGS_PATH, CLASSIFICATION_THRESHOLD, CLUSTER_SAVE_PATH)
