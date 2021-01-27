import shutil

from input_output_logic import *
import torch

from functools import reduce
from itertools import count

from timeit import default_timer
import logging
logging.basicConfig(level=logging.INFO)


CLASSIFICATION_THRESHOLD = 0.95  # 0.53  # OR 0.73 cf. Bijl - A comparison of clustering algorithms for face clustering
RECLUSTERING_THRESHOLD = 2 * CLASSIFICATION_THRESHOLD  # max dist to cluster center before an embed. gets own cluster

MAX_CLUSTER_SIZE = 64
_MAX_NUM_TOTAL_COMPS = 1000
MAX_NUM_CLUSTER_COMPS = _MAX_NUM_TOTAL_COMPS // MAX_CLUSTER_SIZE  # maximum number of clusters to compute distance to

CLUSTERS_PATH = 'stored_clusters'
EMBEDDINGS_PATH = 'stored_embeddings'


# design decision re. splitting: check for cluster size first, because greater efficiency (short-circuit), although if
# both cluster-size and too far from center apply, it would be better to start reclustering with new embedding.

# TODO: remove
_NUM_EMBEDDINGS_TO_CLASSIFY = 100


# TODO: Add proper comments!
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
        return list(self.embeddings.values())

    # def get_embedding_id(self, embedding):
    #     # TODO: implement(?)
    #     return ...

    def get_size(self):
        return len(self.embeddings)

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
        embeddings = list(load_tensors(path_to_cluster, from_path=True))
        cluster = cls(embeddings)
        if cluster_id is not None:
            cluster.cluster_id = cluster_id
        return cluster

    @staticmethod
    def sum_embeddings(embeddings):
        return reduce(torch.add, embeddings)


def cluster_embeddings(embeddings, classification_threshold, max_num_cluster_comps, reclustering_threshold=None,
                       max_cluster_size=None, cluster_save_path=None):
    """
    Build clusters from face embeddings stored in the given path using the specified classification threshold.
    (Currently handled as: All embeddings closer than the distance given by the classification threshold are placed in
    the same cluster. If cluster_save_path is set, store the resulting clusters as directories in the given path.

    :param embeddings: Either a path (string) to the directory wherein embeddings are located or an iterable containing
    the embeddings
    :param classification_threshold:
    :param max_num_cluster_comps:
    :param reclustering_threshold:
    :param max_cluster_size:
    :param cluster_save_path:
    :return:
    """
    embeddings_loader = load_tensors(embeddings, yield_paths=True, from_path=isinstance(embeddings, str))
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
    counter_vals = range(2, _NUM_EMBEDDINGS_TO_CLASSIFY + 1) if _NUM_EMBEDDINGS_TO_CLASSIFY >= 0 else count(2)
    for counter, (embedding_file_path, new_embedding) in zip(counter_vals, embeddings_loader):  # enumerate(embeddings_loader, start=2):
        if counter % 100 == 0:
            logging.info(f' --- Current embedding number: {counter}')

        # sort clusters by distance from their center to embedding; only consider closest clusters
        clusters_by_center_dist = sorted(clusters, key=lambda cluster: cluster.compute_dist_to_center(new_embedding))
        closest_clusters = clusters_by_center_dist[:max_num_cluster_comps]

        shortest_emb_dist, closest_cluster = find_closest_cluster_to_embedding(closest_clusters, new_embedding)
        embedding_file_name = os.path.split(embedding_file_path)[-1]

        if shortest_emb_dist <= classification_threshold:
            closest_cluster.add_embedding(new_embedding, embedding_file_name)
            start_embeddings = _get_embs_too_far_from_center(closest_cluster, reclustering_threshold)
            if len(start_embeddings) > 0 or _is_cluster_too_big(closest_cluster, max_cluster_size):
                recluster_without_splitting(closest_cluster, clusters, classification_threshold, max_num_cluster_comps,
                                            start_embeddings=[new_embedding])
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
    else:
        return clusters


def _is_cluster_too_big(cluster, max_cluster_size):
    return max_cluster_size is not None and cluster.get_size() >= max_cluster_size


def _get_embs_too_far_from_center(cluster, max_dist):
    """
    :param max_dist: Maximum distance to from center allowed
    """
    if max_dist is None:
        return []
    return list(filter(lambda emb: cluster.compute_dist_to_center(emb) > max_dist,
                       cluster.get_embeddings()))


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
        embeddings = cluster.get_embeddings()
        emb_dists = map(lambda cluster_emb: compute_dist(embedding, cluster_emb), embeddings)
        min_cluster_emb_dist = min(emb_dists)
        if min_cluster_emb_dist < shortest_emb_dist:
            shortest_emb_dist = min_cluster_emb_dist
            closest_cluster = cluster
    if return_dist:
        return shortest_emb_dist, closest_cluster
    return closest_cluster


def recluster_without_splitting(closest_cluster, clusters, classification_threshold, max_num_cluster_comps,
                                start_embeddings=None):
    """
    Recluster given cluster without further (indirectly recursive) splitting.

    :param start_embeddings: Embeddings which to cluster first
    """
    embeddings = closest_cluster.get_embeddings()
    if start_embeddings is not None:
        embeddings = start_embeddings + embeddings
    new_clusters = _recluster_without_splitting_worker(embeddings, classification_threshold,
                                                       max_num_cluster_comps)
    clusters.extend(new_clusters)
    clusters.remove(closest_cluster)


def _recluster_without_splitting_worker(embeddings, threshold, max_num_cluster_comp):
    # TODO: Start with newly added embedding!
    return cluster_embeddings(embeddings, threshold, max_num_cluster_comp)


# ------- HELPERS -------


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


def remove_directory_trees(dir_tree_paths):
    for dir_tree_path in dir_tree_paths:
        shutil.rmtree(dir_tree_path, onerror=_handle_tree_removal_errors)


def _handle_tree_removal_errors(function, path, excinfo):
    exc_type, _, traceback = excinfo
    if exc_type is FileNotFoundError:
        log_error(f'File or directory at {path} not found')
    elif exc_type is PermissionError:
        log_error(f'No permission to remove file or directory at {path}')
    elif exc_type is OSError:
        log_error(f'The following error occurred for the file or directory at {path}:' '\n' + str(traceback))
    else:
        raise


if __name__ == '__main__':
    remove_directory_trees([CLUSTERS_PATH])
    cluster_embeddings(EMBEDDINGS_PATH, CLASSIFICATION_THRESHOLD, MAX_NUM_CLUSTER_COMPS, RECLUSTERING_THRESHOLD,
                       MAX_CLUSTER_SIZE, CLUSTERS_PATH)
