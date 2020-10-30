import os

from itertools import combinations_with_replacement

from core_algorithm import _are_same_person


CLUSTERS_PATH = 'stored_clusters'

# cf. Bijl - A comparison of clustering algorithms for face clustering


def compute_f_measure(clusters_path):
    num_true_positives = count_true_positives(clusters_path)
    precision = compute_pairwise_precision(clusters_path, num_true_positives)
    recall = compute_pairwise_recall(clusters_path, num_true_positives)
    return 2 * precision * recall / (precision + recall)


def compute_pairwise_precision(clusters_path, num_true_positives=None):
    """Fraction of faces correctly clustered together of all faces clustered together"""
    # TODO: What does that measure mean??
    if num_true_positives is None:
        num_true_positives = count_true_positives(clusters_path)
    num_false_positives = count_false_positives(clusters_path)
    return num_true_positives / (num_true_positives + num_false_positives)


def compute_pairwise_recall(clusters_path, num_true_positives=None):
    # TODO: Correct docstring!
    # TODO: What does that measure mean??
    """Fraction of faces correctly clustered together of all faces clustered together"""
    if num_true_positives is None:
        num_true_positives = count_true_positives(clusters_path)
    num_false_negatives = count_false_negatives(clusters_path)
    return num_true_positives / (num_true_positives + num_false_negatives)


def count_true_positives(clusters_path):
    """Number of face pairs correctly clustered to same cluster"""
    total_true_positives = 0
    for cluster_name in os.listdir(clusters_path):
        cluster_embeddings_path = os.path.join(clusters_path, cluster_name)
        cluster_embeddings = os.listdir(cluster_embeddings_path)
        embedding_pairs = combinations_with_replacement(cluster_embeddings, 2)
        cluster_true_positives = sum(map(lambda pair: _are_same_person(*pair),
                                         embedding_pairs))
        total_true_positives += cluster_true_positives
    return total_true_positives


def count_false_positives(clusters_path):
    """Number of face pairs incorrectly clustered to same cluster"""
    return 0


def count_false_negatives(clusters_path):
    """Number of face pairs incorrectly clustered to different clusters"""
    return 0


# TODO: Useful??
def _compute_dummy_metrics(clusters_path):
    """
    Compute f_measure of the two simplest clusterings:
        1. every embedding gets its own cluster (#clusters = #embeddings)
        2. every embedding placed in same cluster (#clusters = 1)
    """
    # TODO: Only number of embeddings needed, provide directly??
    pass


if __name__ == '__main__':
    f_measure = compute_f_measure(CLUSTERS_PATH)
    print(f'The f-measure of the current clustering is: {f_measure}')
