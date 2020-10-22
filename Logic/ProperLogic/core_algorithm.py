from input_output_logic import *
import torch

from functools import reduce

import logging


CLASSIFICATION_THRESHOLD = 0.8
YIELD_PATHS = False
_NUM_EMBEDDINGS_TO_CLASSIFY = 20


class Cluster:

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

    @staticmethod
    def sum_embeddings(embeddings):
        return reduce(torch.add, embeddings)

    # def compute_avg_dist_to_center(self, dim=None):
    #     pass


def main_algorithm(classification_threshold, yield_paths=True):
    embeddings_loader = load_embeddings_from_path(TENSORS_PATH, yield_paths)
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
    print('byebye!')


def log_error(msg):
    logging.error('Error: ' + msg)


if __name__ == '__main__':
    main_algorithm(CLASSIFICATION_THRESHOLD, YIELD_PATHS)
