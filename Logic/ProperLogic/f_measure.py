import os

from itertools import combinations, combinations_with_replacement, product

from core_algorithm import _are_same_person

import logging
logging.basicConfig(level=logging.INFO)

import time

CLUSTERS_PATH = 'stored_clusters'
SAVE_RESULTS = True

# cf. Bijl - A comparison of clustering algorithms for face clustering


# TODO: For future: how to do this more elegantly? write/log to string buffer or sth?
TEMP_OUTPUT_DICT = {}


def compute_f_measure(clusters_path):
    global TEMP_OUTPUT_DICT
    num_true_positives = count_true_positives(clusters_path)
    precision = compute_pairwise_precision(clusters_path, num_true_positives)
    recall = compute_pairwise_recall(clusters_path, num_true_positives)
    TEMP_OUTPUT_DICT['precision'] = precision
    TEMP_OUTPUT_DICT['recall'] = recall
    return 2 * precision * recall / (precision + recall)


def compute_pairwise_precision(clusters_path, num_true_positives=None):
    global TEMP_OUTPUT_DICT
    """Fraction of faces correctly clustered together of all faces clustered together"""
    # intuition: Of all faces *placed* in same cluster(s), how many really *belonged* there (together)?
    # best when: clusters small
    if num_true_positives is None:
        num_true_positives = count_true_positives(clusters_path)
    num_false_positives = count_false_positives(clusters_path)
    TEMP_OUTPUT_DICT['true positives'] = num_true_positives
    TEMP_OUTPUT_DICT['false positives'] = num_false_positives
    return num_true_positives / (num_true_positives + num_false_positives)


def compute_pairwise_recall(clusters_path, num_true_positives=None):
    global TEMP_OUTPUT_DICT
    """Fraction of faces correctly clustered together of all faces belonging together"""
    # intuition: Of all faces which *belonged* in same cluster(s), how many were actually *placed* there (together)?
    # best when: clusters big
    if num_true_positives is None:
        num_true_positives = count_true_positives(clusters_path)
    num_false_negatives = count_false_negatives(clusters_path)
    TEMP_OUTPUT_DICT['false negatives'] = num_false_negatives
    return num_true_positives / (num_true_positives + num_false_negatives)


# TODO: Does order matter??? I.e. is (f1, f2) != (f2, f1)? Does this matter for the evaluation??
# TODO: Parallelize the counting? Is this important? ---> only if slow or cumbersome!
def count_true_positives(clusters_path):
    """Number of face pairs correctly clustered to same cluster"""
    return _count_positives(clusters_path, True)


def count_false_positives(clusters_path):
    """Number of face pairs incorrectly clustered to same cluster"""
    return _count_positives(clusters_path, False)


def _count_positives(clusters_path, type_of_positives):
    """
    ...
    :param clusters_path: Path to the clusters, each containing embeddings
    :param type_of_positives: Boolean(!) indicating whether true or false positives are to be returned.
    :return: Number of
    """
    def does_match(pair):
        # Count iff result of check (yes/no) is same as wanted type (true/false positives)
        return _are_same_person(*pair) is type_of_positives

    clusters_embedding_pairs = _get_intra_clusters_embedding_pairs(clusters_path)
    total_positives = 0
    for embedding_pairs in clusters_embedding_pairs:
        cluster_positives = sum(map(does_match, embedding_pairs))
        total_positives += cluster_positives
    return total_positives


def count_false_negatives(clusters_path):
    """Number of face pairs incorrectly clustered to different clusters"""
    """
    ...
    :param clusters_path: Path to the clusters, each containing embeddings
    :param type_of_positives: Boolean(!) indicating whether true or false positives are to be returned.
    :return: Number of
    """
    clusters_embedding_pairs = _get_inter_clusters_embedding_pairs(clusters_path)
    total_positives = 0
    for embedding_pairs in clusters_embedding_pairs:
        # if count % 10000 == 0:
        #     logging.info(f'--- --- embeddings iteration: {count}')
        cluster_positives = sum(map(lambda pair: _are_same_person(*pair), embedding_pairs))
        total_positives += cluster_positives
    return total_positives


# TODO: Refactor
def _get_intra_clusters_embedding_pairs(clusters_path):
    for cluster_name in os.listdir(clusters_path):
        cluster_embeddings_path = os.path.join(clusters_path, cluster_name)
        cluster_embeddings = os.listdir(cluster_embeddings_path)
        embedding_pairs = combinations_with_replacement(cluster_embeddings, 2)
        yield embedding_pairs


# TODO: Refactor
def _get_inter_clusters_embedding_pairs(clusters_path):
    cluster_pairs = combinations(os.listdir(clusters_path), 2)
    logging.info('STARTING INTER-CLUSTERS ITERATIONS')
    for count, (cluster1_name, cluster2_name) in enumerate(cluster_pairs):
        if count % 10000 == 0:
            logging.info(f' --- cluster iteration: {count}')
        cluster1_embeddings_path = os.path.join(clusters_path, cluster1_name)
        cluster2_embeddings_path = os.path.join(clusters_path, cluster2_name)
        cluster1_embeddings = os.listdir(cluster1_embeddings_path)
        cluster2_embeddings = os.listdir(cluster2_embeddings_path)

        embedding_pairs = product(cluster1_embeddings, cluster2_embeddings)
        yield embedding_pairs


# # TODO: Useful??
# def _compute_dummy_metrics(clusters_path):
#     """
#     Compute f_measure of the two simplest clusterings:
#         1. every embedding gets its own cluster (#clusters = #embeddings)
#         2. every embedding placed in same cluster (#clusters = 1)
#     """
#     # TODO: Only number of embeddings needed, provide directly??
#     pass


if __name__ == '__main__':
    if not SAVE_RESULTS:
        ans = input("Really don't save the results? Press Enter without entering anything to abort.\n")
        if not ans:
            exit()
        print()

    f_measure = compute_f_measure(CLUSTERS_PATH)
    TEMP_OUTPUT_DICT['f-measure'] = f_measure
    num_embeddings = len(os.listdir('stored_clusters'))
    # num_pairs = sum i=0...num_embeddings-1 {i} = n (n-1) / 2
    # = len(list(combinations(range(n), 2)))
    # = 2293011
    num_pairs = num_embeddings * (num_embeddings - 1) / 2
    num_true_negatives = num_pairs - sum(value for value in TEMP_OUTPUT_DICT.values() if str(value).isdigit())
    TEMP_OUTPUT_DICT['true negatives'] = num_true_negatives

    if SAVE_RESULTS:
        file_name = f'results_{round(time.time())}.txt'
        with open(file_name, 'w') as file:
            output = '\n'.join(f'{key}: {value}' for key, value in TEMP_OUTPUT_DICT.items())
            file.write(output)
