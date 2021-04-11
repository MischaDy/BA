import os
from functools import partial
from typing import Union, Tuple

from Logic.ProperLogic.cluster_modules.cluster import Cluster
from Logic.ProperLogic.cluster_modules.cluster_dict import ClusterDict
from Logic.ProperLogic.database_modules.database_logic import DBManager
from misc_helpers import remove_items, starfilterfalse

from itertools import combinations

import logging
logging.basicConfig(level=logging.INFO)


# design decision re. splitting: check for cluster size first, because greater efficiency (short-circuit), although if
# both cluster-size and too far from center apply, it would be better to start reclustering with new embedding.


CLUSTERS_PATH = 'stored_clusters'
EMBEDDINGS_PATH = 'stored_embeddings'


# TODO: Test, that cluster-split works and that params are ok!

class CoreAlgorithm:
    path_to_central_db = os.path.join(DBManager.db_files_path, DBManager.central_db_file_name)
    path_to_local_db = os.path.join(DBManager.db_files_path, DBManager.local_db_file_name)

    # 0.53  # OR 0.73 cf. Bijl - A comparison of clustering algorithms for face clustering
    classification_threshold = 0.73
    # TODO: Is that a sensible value?
    reclustering_threshold = 2 * classification_threshold  # max dist to cluster center before an emb. gets own cluster

    max_cluster_size = 100
    max_num_total_comps = 1000
    max_num_cluster_comps = max_num_total_comps // max_cluster_size  # maximum number of clusters to compute distance to

    # TODO: remove
    num_embeddings_to_classify = -1

    @classmethod
    def cluster_embeddings(cls, embeddings, embeddings_ids=None, existing_clusters_dict=None, final_clusters_only=True):
        """
        Build clusters from face embeddings stored in the given path using the specified classification threshold.
        (Currently handled as: All embeddings closer than the distance given by the classification threshold are placed
        in the same cluster. If cluster_save_path is set, store the resulting clusters as directories in the given path.

        :param embeddings: Iterable containing the embeddings. It embeddings_ids is None, must consist of
        (id, embedding)-pairs
        :param embeddings_ids: Ordered iterable with the embedding ids. Must be at least as long as embeddings.
        :param existing_clusters_dict:
        :param final_clusters_only: If true, only the final iterable of clusters is returned. Otherwise, return that
        final iterable, as well as a list of modified/newly created and deleted clusters
        :return:
        """
        # TODO: Improve efficiency?
        # TODO: Allow embeddings_ids to be none? Get next id via DB query?
        # TODO: Allow embeddings_ids to be shorter than embeddings and 'fill up' remaining ids?
        if not embeddings:
            if final_clusters_only:
                return []
            return [], [], []

        if embeddings_ids is None:
            embeddings_with_ids = embeddings
        else:
            if len(embeddings) > len(embeddings_ids):
                raise ValueError(f'Too few ids for embeddings ({len(embeddings_ids)} passed, but {len(embeddings)}'
                                 f' needed)')
            embeddings_with_ids = zip(embeddings_ids, embeddings)

        if existing_clusters_dict is None:
            existing_clusters_dict = ClusterDict()
        else:
            # TODO: Improve efficiency? (better algorithm)
            # Don't iterate over embeddings in existing clusters
            def exists_in_any_cluster(emb_id, _):
                return existing_clusters_dict.any_cluster_with_emb(emb_id)

            embeddings_with_ids = starfilterfalse(exists_in_any_cluster, embeddings_with_ids)
        cluster_dict = existing_clusters_dict
        modified_clusters_ids, removed_clusters_ids = set(), set()

        max_existing_id = existing_clusters_dict.get_max_id()
        max_db_id = DBManager.get_max_cluster_id()
        next_cluster_id = max(max_existing_id, max_db_id) + 1

        # counter_vals = (range(2, cls.num_embeddings_to_classify + 1) if cls.num_embeddings_to_classify >= 0
        #                 else count(2))
        for embedding_id, new_embedding in embeddings_with_ids:

            # sort clusters by distance from their center to embedding; only consider closest clusters
            clusters_by_center_dist = sorted(cluster_dict.get_clusters(),
                                             key=lambda cluster: cluster.compute_dist_to_center(new_embedding))
            closest_clusters = clusters_by_center_dist[:cls.max_num_cluster_comps]

            # find cluster containing the closest embedding to new_embedding
            shortest_emb_dist, closest_cluster = cls.find_closest_cluster_to_embedding(closest_clusters, new_embedding)

            if shortest_emb_dist <= cls.classification_threshold:
                closest_cluster.add_embedding(new_embedding, embedding_id)
                modified_clusters_ids.add(closest_cluster.cluster_id)

                is_cluster_too_big = cls.is_cluster_too_big(closest_cluster)
                if is_cluster_too_big or cls.exists_emb_too_far_from_center(closest_cluster):
                    new_clusters = cls.split_cluster(closest_cluster)
                    cluster_dict.remove_cluster(closest_cluster)
                    removed_clusters_ids.add(closest_cluster.cluster_id)
                    cluster_dict.add_clusters(new_clusters)
                    for new_cluster in new_clusters:
                        modified_clusters_ids.add(new_cluster.cluster_id)
            else:
                new_cluster = Cluster(next_cluster_id, [new_embedding], [embedding_id])
                next_cluster_id += 1
                cluster_dict.add_cluster(new_cluster)
                modified_clusters_ids.add(new_cluster.cluster_id)

        if final_clusters_only:
            return cluster_dict
        modified_clusters = cluster_dict.get_clusters_by_ids(modified_clusters_ids)
        removed_clusters = cluster_dict.get_clusters_by_ids(removed_clusters_ids)
        return cluster_dict, ClusterDict(modified_clusters), ClusterDict(removed_clusters)

    @classmethod
    def is_cluster_too_big(cls, cluster):
        return cls.max_cluster_size is not None and cluster.get_size() >= cls.max_cluster_size

    @classmethod
    def exists_emb_too_far_from_center(cls, cluster):
        if cls.reclustering_threshold is None:
            return False

        def is_too_far_from_center(emb):
            return cluster.compute_dist_to_center(emb) > cls.reclustering_threshold
        return any(map(is_too_far_from_center, cluster.get_embeddings()))

    @classmethod
    def find_closest_cluster_to_embedding(cls, clusters, embedding, return_dist=True) -> Union[Cluster,
                                                                                               Tuple[float, Cluster]]:
        """
        Determine closest cluster to current embedding, i.e. the one which stores the closest embedding to
        the current embedding.

        :param clusters: Iterable of clusters from which the closest one should be picked.
        :param embedding: The embedding to find closest cluster to
        :param return_dist: If True, distance to closest cluster embedding is also returned.
        :return: The cluster storing the embedding which is closest to given embedding
        """
        # TODO: Improve efficiency!
        shortest_emb_dist = float('inf')
        closest_cluster = None
        for cluster in clusters:
            min_cluster_emb_dist = min(map(partial(Cluster.compute_dist, embedding),
                                           cluster.get_embeddings()))
            if min_cluster_emb_dist < shortest_emb_dist:
                shortest_emb_dist = min_cluster_emb_dist
                closest_cluster = cluster
        if return_dist:
            return shortest_emb_dist, closest_cluster
        return closest_cluster

    @classmethod
    def split_cluster(cls, cluster_to_split):
        """
        Split cluster into two new clusters as follows:
        1. Find two embeddings e1, e2 in the cluster with the greatest distance between them.
        2. Create a new cluster C1, C2 for each of the two.
        3. For each embedding e of the remaining embeddings:
               Add e to the cluster (C1 or C2) whose center is closer to it.

        The given cluster must contain at least 2 embeddings.

        :param cluster_to_split: Cluster to be split
        :return: Two new clusters containing embeddings of old one
        """
        # TODO: Improve efficiency!
        embeddings = list(cluster_to_split.get_embeddings())
        cluster_start_emb1, cluster_start_emb2 = cls.find_most_distant_embeddings(embeddings)
        remove_items(embeddings, [cluster_start_emb1, cluster_start_emb2])
        label = cluster_to_split.label

        max_cluster_id = DBManager.get_max_cluster_id()
        new_cluster1_id, new_cluster2_id = max_cluster_id + 1, max_cluster_id + 2
        new_cluster1, new_cluster2 = (Cluster(new_cluster1_id, cluster_start_emb1, label=label),
                                      Cluster(new_cluster2_id, cluster_start_emb2, label=label))
        for embedding in embeddings:
            dist_to_cluster1 = new_cluster1.compute_dist_to_center(embedding)
            dist_to_cluster2 = new_cluster2.compute_dist_to_center(embedding)
            if dist_to_cluster1 < dist_to_cluster2:
                new_cluster1.add_embedding(embedding)
            else:
                new_cluster2.add_embedding(embedding)
        return new_cluster1, new_cluster2

    @staticmethod
    def find_most_distant_embeddings(embeddings):
        """
        Return the two embeddings which have the greatest distance between them.
        :param embeddings:
        :return:
        """
        # TODO: Improve efficiency?
        if not len(embeddings) > 1:
            raise ValueError("'embeddings' must contain at least 2 embeddings")
        # max_dist = -1
        # max_dist_embs = (None, None)
        # for ind1, emb1 in enumerate(embeddings):
        #     for emb2 in embeddings[ind1+1:]:
        #         cur_dist = Cluster.compute_dist(emb1, emb2)
        #         if cur_dist > max_dist:
        #             max_dist = cur_dist
        #             max_dist_embs = (emb1, emb2)
        embs_pairs = combinations(embeddings, r=2)
        return max(embs_pairs, key=Cluster.compute_dist)
