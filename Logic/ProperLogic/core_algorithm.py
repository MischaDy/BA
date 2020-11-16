from input_output_logic import *
import torch

from functools import reduce
from itertools import count

from timeit import default_timer
import logging
logging.basicConfig(level=logging.INFO)


MAX_NUM_CLUSTER_COMP = 10  # maximum number of clusters to compute distance to
CLASSIFICATION_THRESHOLD = 0.6  # OR 0.73 cf. Bijl - A comparison of clustering algorithms for face clustering
#_NUM_EMBEDDINGS_TO_CLASSIFY = -1

CLUSTERS_PATH = 'stored_clusters'
EMBEDDINGS_PATH = 'stored_embeddings'


# TODO: Proper comments!
# TODO: Store thumbnail imgs WITH the tensors
# TODO: Make embedding id 'proper' id?? ---> Also in input_output_logic!

# TODO: Decide on exact way to split cluster??
# TODO: (Which) limit to put on cluster size before splitting?

# TODO: Algorithm sensitive to input order(?)!! How to fix? Line-sweep or similar? Fixable at all? Relevant?


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
                embeddings_ids = count(1)
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

    # def get_embedding_id(self, embedding):
    #     # TODO: implement(?)
    #     return ...

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
        return compute_dist(self.center_point, embedding)

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


def main_algorithm(embeddings_path, classification_threshold, max_num_cluster_comp, cluster_save_path=None):
    """
    Build clusters from face embeddings stored in the given path using the specified classification threshold.
    (Currently handled as: All embeddings closer than the distance given by the classification threshold are placed in
    the same cluster. If cluster_save_path is set, store the resulting clusters as directories in the given path.

    :param max_num_cluster_comp:
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
    logging.info('START iteration over embeddings')
    time1 = default_timer()
    # counter_vals = range(2, _NUM_EMBEDDINGS_TO_CLASSIFY + 1) if _NUM_EMBEDDINGS_TO_CLASSIFY >= 0 else count(2)
    # zip(counter_vals, embeddings_loader):
    for counter, (embedding_file_path, new_embedding) in enumerate(embeddings_loader, start=2):
        if counter % 100 == 0:
            logging.info(f' --- Current embedding number: {counter}')

        # sort clusters by distance from their center to embedding; only consider closest clusters
        clusters_by_center_dist = sorted(clusters, key=lambda cluster: cluster.compute_dist_to_center(new_embedding))
        closest_clusters = clusters_by_center_dist[:max_num_cluster_comp]

        shortest_emb_dist, closest_cluster = find_closest_cluster_to_embedding(closest_clusters, new_embedding)
        embedding_file_name = os.path.split(embedding_file_path)[-1]

        if shortest_emb_dist <= classification_threshold:
            closest_cluster.add_embedding(new_embedding, embedding_file_name)
        else:
            clusters.append(Cluster([new_embedding], [embedding_file_name]))

    logging.info(f' --- Last embedding number: {counter}')
    logging.info(f'END iteration over embeddings')
    logging.info(f'Time spent on embeddings: {default_timer() - time1}')

    if cluster_save_path is not None:
        time1 = default_timer()
        logging.info('\nSTART cluster saving')
        for counter, cluster in enumerate(clusters, start=1):
            if counter % 100 == 0:
                logging.info(f' --- Current cluster number: {counter}')
            cluster.save_cluster(cluster_save_path)
        logging.info(f' --- Last cluster number: {counter}')
        logging.info(f'END cluster saving')
        logging.info(f'Time spent on saving clusters: {default_timer() - time1}')


def find_closest_cluster_to_embedding(clusters, embedding, return_dist=True):
    """
    Determine closest cluster to current embedding, i.e. the one which stores the closest embedding to
    the current embedding.

    :param clusters: Iterable of clusters from which the closest one should be picked.
    :param embedding: The embedding to find closest cluster to
    :param return_dist: If True, distance to closest cluster embedding is also returned.
    :return: The cluster storing the embedding which is closest to given embedding
    """
    shortest_emb_dist = float('inf')
    closest_cluster = None
    for cluster in clusters:
        cluster_embeddings = cluster.get_embeddings()
        emb_dists = map(lambda cluster_emb: compute_dist(embedding, cluster_emb), cluster_embeddings)
        min_cluster_emb_dist = min(emb_dists)
        if min_cluster_emb_dist < shortest_emb_dist:
            shortest_emb_dist = min_cluster_emb_dist
            closest_cluster = cluster
    if return_dist:
        return shortest_emb_dist, closest_cluster
    return closest_cluster


def log_error(msg):
    logging.error('Error: ' + msg)


def _are_same_person(embedding_name1, embedding_name2):
    # TODO: Don't assume underscore based naming with digits at end as only difference!?
    person1_numbered_name, _ = os.path.splitext(embedding_name1)
    person2_numbered_name, _ = os.path.splitext(embedding_name2)
    person1_name = _rstrip_underscored_part(person1_numbered_name)
    person2_name = _rstrip_underscored_part(person2_numbered_name)
    return person1_name == person2_name


def _rstrip_underscored_part(string):
    """Remove part after rightmost underscore in string if such a part exists."""
    underscore_ind = string.rfind('_')
    if underscore_ind != -1:
        return string[:underscore_ind]
    return string


def compute_dist(embedding1, embedding2):
    return float(torch.dist(embedding1, embedding2))


if __name__ == '__main__':
    main_algorithm(EMBEDDINGS_PATH, CLASSIFICATION_THRESHOLD, MAX_NUM_CLUSTER_COMP, CLUSTERS_PATH)
