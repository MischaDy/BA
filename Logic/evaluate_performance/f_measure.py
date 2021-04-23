import os
import time

from itertools import combinations, combinations_with_replacement, product

import logging

from Logic.ProperLogic.misc_helpers import get_multiple

logging.basicConfig(level=logging.INFO)

# TODO: Randomize order of input images/embeddings, to check algorithmic stability (of clustering quality)!

# SAVE_RESULTS = True
# SAVE_PATH = 'results'
# INTRACLUSTER_ITERATIONS_PROGRESS = 1000
INTERCLUSTER_ITERATIONS_PROGRESS = 200000


# cf. Bijl - A comparison of clustering algorithms for face clustering

class FMeasureComputation:
    def compute_f_measure(self, clusters, emb_id_to_name_dict, output_dict):
        num_true_positives = self.count_true_positives(clusters, emb_id_to_name_dict)
        precision = self.compute_pairwise_precision(clusters, emb_id_to_name_dict, output_dict, num_true_positives)
        recall = self.compute_pairwise_recall(clusters, emb_id_to_name_dict, output_dict, num_true_positives)
        output_dict['precision'] = precision
        output_dict['recall'] = recall
        return 2 * precision * recall / (precision + recall)

    def compute_pairwise_precision(self, clusters, emb_id_to_name_dict, output_dict, num_true_positives=None):
        """Fraction of faces correctly clustered together of all faces clustered together"""
        # intuition: Of all faces *placed* in same cluster(s), how many really *belonged* there (together)?
        # best when: clusters small
        if num_true_positives is None:
            num_true_positives = self.count_true_positives(clusters, emb_id_to_name_dict)
        num_false_positives = self.count_false_positives(clusters, emb_id_to_name_dict)
        output_dict['true positives'] = num_true_positives
        output_dict['false positives'] = num_false_positives
        return num_true_positives / (num_true_positives + num_false_positives)

    def compute_pairwise_recall(self, clusters, emb_id_to_name_dict, output_dict, num_true_positives=None):
        """Fraction of faces correctly clustered together of all faces belonging together"""
        # intuition: Of all faces which *belonged* in same cluster(s), how many were actually *placed* there (together)?
        # best when: clusters big
        if num_true_positives is None:
            num_true_positives = self.count_true_positives(clusters, emb_id_to_name_dict)
        num_false_negatives = self.count_false_negatives(clusters, emb_id_to_name_dict)
        output_dict['false negatives'] = num_false_negatives
        return num_true_positives / (num_true_positives + num_false_negatives)

    # TODO: Does order matter??? I.e. is (f1, f2) != (f2, f1)? Does this matter for the evaluation??
    # TODO: Parallelize the counting? Is this important? ---> only if slow or cumbersome!
    def count_true_positives(self, clusters, emb_id_to_name_dict):
        """Number of face pairs correctly clustered to same cluster"""
        return self._count_positives(clusters, emb_id_to_name_dict, True)

    def count_false_positives(self, clusters, emb_id_to_name_dict):
        """Number of face pairs incorrectly clustered to same cluster"""
        return self._count_positives(clusters, emb_id_to_name_dict, False)

    def _count_positives(self, clusters, emb_id_to_name_dict, type_of_positives):
        """
        ...
        :param type_of_positives: Boolean(!) indicating whether true or false positives are to be returned.
        :return: Number of
        """

        def does_match(self, emb_id_pair):
            # Count iff result of check (yes/no) is same as wanted type (true/false positives)
            return self.are_same_person_func(*emb_id_pair, emb_id_to_name_dict) is type_of_positives

        # TODO: output one level to unflat?
        clusters_embedding_id_pairs = self._get_intra_clusters_embedding_id_pairs(clusters)
        total_positives = 0
        for embedding_id_pairs in clusters_embedding_id_pairs:
            cluster_positives = sum(map(does_match, embedding_id_pairs))
            total_positives += cluster_positives
        return total_positives

    def count_false_negatives(self, clusters, emb_id_to_name_dict):
        """
        Number of face pairs incorrectly clustered to different clusters

        :return: Number of
        """

        def does_match(self, emb_id_pair):
            # Count iff result of check (yes/no) is same as wanted type (true/false positives)
            return self.are_same_person_func(*emb_id_pair, emb_id_to_name_dict)

        clusters_embedding_pairs = self._get_inter_clusters_embedding_id_pairs(clusters)
        total_positives = 0
        for count, embedding_pairs in enumerate(clusters_embedding_pairs, start=1):
            # if count % INTERCLUSTER_ITERATIONS_PROGRESS == 0:
            #     logging.info(f'--- --- embeddings iteration: {count}')
            cluster_positives = sum(map(does_match, embedding_pairs))
            total_positives += cluster_positives
        return total_positives

    @staticmethod
    def _get_intra_clusters_embedding_id_pairs(clusters):
        # TODO: Refactor
        for cluster in clusters:
            cluster_embeddings_ids = cluster.get_embeddings_ids()
            embedding_id_pairs = combinations_with_replacement(cluster_embeddings_ids, 2)
            yield embedding_id_pairs

    @staticmethod
    def _get_inter_clusters_embedding_id_pairs(clusters):
        # TODO: Refactor
        cluster_pairs = combinations(clusters, 2)
        logging.info('STARTING INTER-CLUSTERS ITERATIONS')
        for count, (cluster1, cluster2) in enumerate(cluster_pairs):
            if count % INTERCLUSTER_ITERATIONS_PROGRESS == 0:
                logging.info(f' --- cluster iteration: {count}')
            cluster1_embeddings_ids = cluster1.get_embeddings_ids()
            cluster2_embeddings_ids = cluster2.get_embeddings_ids()

            embeddings_ids_pairs = product(cluster1_embeddings_ids, cluster2_embeddings_ids)
            yield embeddings_ids_pairs


def main(clusters, emb_id_to_name_dict, are_same_person_func, save_results, save_path, save_file_name_postfix=''):
    if not save_results:
        ans = input("Really don't save the results? Press Enter without entering anything to abort function.\n")
        if not ans:
            exit()
        print()

    output_dict = {}
    f_measure_comp = FMeasureComputation(are_same_person_func)
    f_measure = f_measure_comp.compute_f_measure(clusters, emb_id_to_name_dict, output_dict)
    output_dict['f-measure'] = f_measure
    num_embeddings = len(emb_id_to_name_dict)
    # num_pairs = sum i=0...num_embeddings-1 {i} = n (n-1) / 2
    # = len(list(combinations(range(n), 2)))
    # = 2293011
    num_pairs = num_embeddings * (num_embeddings + 1) / 2
    other_result_types = ['true positives', 'false positives', 'false negatives']
    num_true_negatives = num_pairs - sum(get_multiple(output_dict, other_result_types))
    output_dict['true negatives'] = num_true_negatives

    if save_results:
        save_f_measure_result(save_path, output_dict, save_file_name_postfix)


def save_f_measure_result(save_path, output_dict, save_file_name_postfix):
    if save_file_name_postfix and not save_file_name_postfix.startswith('_'):
        save_file_name_postfix = '___' + save_file_name_postfix
    file_name = f'results_{round(time.time())}{save_file_name_postfix}.txt'
    file_path = os.path.join(save_path, file_name)
    with open(file_path, 'w') as file:
        output = '\n'.join(f'{key}: {value}' for key, value in output_dict.items())
        file.write(output)

# # TODO: Useful??
# def _compute_dummy_metrics(clusters_path):
#     """
#     Compute f_measure of the two simplest clusterings:
#         1. every embedding gets its own cluster (#clusters = #embeddings)
#         2. every embedding placed in same cluster (#clusters = 1)
#     """
#     # TODO: Only number of embeddings needed, provide directly??
#     pass


# if __name__ == '__main__':
#     main(SAVE_RESULTS, SAVE_PATH)
