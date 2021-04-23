from functools import partial
from typing import Union, Tuple

from Logic.ProperLogic.cluster_modules.cluster import Cluster
from Logic.ProperLogic.cluster_modules.cluster_dict import ClusterDict
from Logic.ProperLogic.database_modules.database_logic import DBManager
from Logic.ProperLogic.misc_helpers import remove_items, starfilterfalse, partition

from itertools import combinations

import logging

from Logic.ProperLogic.helper_classes.sorted_list import SortedList

logging.basicConfig(level=logging.INFO)


# design decision re. splitting: check for cluster size first, because greater efficiency (short-circuit), although if
# both cluster-size and too far from center apply, it would be better to start reclustering with new embedding.


CLUSTERS_PATH = 'stored_clusters'
EMBEDDINGS_PATH = 'stored_embeddings'


# TODO: Test, that cluster-split works and that params are ok!

class CoreAlgorithm:
    def __init__(self, classification_threshold=0.73, r=2, max_cluster_size=100, max_num_total_comps=1000):
        # 0.53  # OR 0.73 cf. Bijl - A comparison of clustering algorithms for face clustering
        # TODO: Is r=2 a sensible value?

        self.classification_threshold = classification_threshold
        # max dist to cluster center before an emb. gets own cluster
        self.reclustering_threshold = r * classification_threshold

        self.max_cluster_size = max_cluster_size
        self.max_num_total_comps = max_num_total_comps
        # maximum number of clusters to compute distance to
        self.max_num_cluster_comps = max_num_total_comps // max_cluster_size

    def cluster_embeddings(self, embeddings, embeddings_ids=None, existing_clusters_dict=None,
                           should_reset_cluster_ids=False, final_clusters_only=True):
        """
        Build clusters from face embeddings stored in the given path using the specified classification threshold.
        (Currently handled as: All embeddings closer than the distance given by the classification threshold are placed
        in the same cluster. If cluster_save_path is set, store the resulting clusters as directories in the given path.

        :param should_reset_cluster_ids:
        :param embeddings: Iterable containing the embeddings. It embeddings_ids is None, must consist of
        (id, embedding)-pairs
        :param embeddings_ids: Ordered iterable with the embedding ids. Must be at least as long as embeddings.
        :param existing_clusters_dict:
        :param final_clusters_only: If true, only the final iterable of clusters is returned. Otherwise, return that
        final iterable, as well as a list of modified/newly created and deleted clusters
        :return:
        """
        # TODO: Allow embeddings_ids to be none? Get next id via DB query?
        # TODO: Allow embeddings_ids to be shorter than embeddings and 'fill up' remaining ids?
        if not embeddings:
            if final_clusters_only:
                return ClusterDict()
            return ClusterDict(), ClusterDict(), ClusterDict()

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
            # Don't iterate over embeddings in existing clusters
            def exists_in_any_cluster(emb_id, _):
                return existing_clusters_dict.any_cluster_with_emb(emb_id)

            embeddings_with_ids = starfilterfalse(exists_in_any_cluster, embeddings_with_ids)

        cluster_dict = existing_clusters_dict
        if should_reset_cluster_ids:
            cluster_dict.reset_ids()
            next_cluster_id = cluster_dict.get_max_id() + 1
        else:
            max_existing_id = cluster_dict.get_max_id()
            max_db_id = DBManager.get_max_cluster_id()
            next_cluster_id = max(max_existing_id, max_db_id) + 1

        modified_clusters_ids, removed_clusters_ids = set(), set()
        for embedding_id, new_embedding in embeddings_with_ids:
            closest_clusters = self.get_closest_clusters(cluster_dict, new_embedding)

            # find cluster containing the closest embedding to new_embedding
            shortest_emb_dist, closest_cluster = self.find_closest_cluster_to_embedding(closest_clusters, new_embedding)

            if shortest_emb_dist <= self.classification_threshold:
                closest_cluster.add_embedding(new_embedding, embedding_id)
                modified_clusters_ids.add(closest_cluster.cluster_id)

                is_cluster_too_big = self.is_cluster_too_big(closest_cluster)
                if is_cluster_too_big or self.exists_emb_too_far_from_center(closest_cluster):
                    new_clusters = self.split_cluster(closest_cluster)
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

    def get_closest_clusters(self, cluster_dict, new_embedding):
        def key(cluster):
            return cluster.compute_dist_to_center(new_embedding)

        clusters = cluster_dict.get_clusters()
        closest_clusters = SortedList(max_size=self.max_num_cluster_comps, key=key)
        for cluster in clusters:
            closest_clusters.add(cluster)
        return closest_clusters

    # def get_closest_clusters(self, cluster_dict, new_embedding):
    #     # sort clusters by distance from their center to embedding; only consider closest clusters
    #     clusters_by_center_dist = sorted(cluster_dict.get_clusters(),
    #                                      key=lambda cluster: cluster.compute_dist_to_center(new_embedding))
    #     closest_clusters = clusters_by_center_dist[:self.max_num_cluster_comps]
    #     return closest_clusters

    def is_cluster_too_big(self, cluster):
        return self.max_cluster_size is not None and cluster.get_size() >= self.max_cluster_size

    def exists_emb_too_far_from_center(self, cluster):
        if self.reclustering_threshold is None:
            return False

        def is_too_far_from_center(emb):
            return cluster.compute_dist_to_center(emb) > self.reclustering_threshold
        return any(map(is_too_far_from_center, cluster.get_embeddings()))

    @classmethod
    def find_closest_cluster_to_embedding(cls, clusters, embedding, return_dist=True) -> Union[Union[Cluster, None],
                                                                                               Tuple[float,
                                                                                                     Union[Cluster,
                                                                                                           None]]]:
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
        embeddings = list(cluster_to_split.get_embeddings())
        cluster_start_emb1, cluster_start_emb2 = cls.find_most_distant_embeddings(embeddings)
        remove_items(embeddings, [cluster_start_emb1, cluster_start_emb2])
        label = cluster_to_split.label

        max_cluster_id = DBManager.get_max_cluster_id()
        new_cluster1_id, new_cluster2_id = max_cluster_id + 1, max_cluster_id + 2
        new_cluster1, new_cluster2 = (Cluster(new_cluster1_id, cluster_start_emb1, label=label),
                                      Cluster(new_cluster2_id, cluster_start_emb2, label=label))

        def is_closer_to_cluster1(emb):
            dist_to_cluster1 = new_cluster1.compute_dist_to_center(emb)
            dist_to_cluster2 = new_cluster2.compute_dist_to_center(emb)
            return dist_to_cluster1 < dist_to_cluster2

        cluster2_embs, cluster1_embs = partition(is_closer_to_cluster1, embeddings)
        new_cluster1.add_embeddings(cluster1_embs)
        new_cluster2.add_embeddings(cluster2_embs)
        return new_cluster1, new_cluster2

    @staticmethod
    def find_most_distant_embeddings(embeddings):
        """
        Return the two embeddings which have the greatest distance between them.
        :param embeddings:
        :return:
        """
        # TODO: Improve efficiency --> No, ok for now
        if not len(embeddings) > 1:
            raise ValueError("'embeddings' must contain at least 2 embeddings")
        embs_pairs = combinations(embeddings, r=2)
        return max(embs_pairs, key=Cluster.compute_dist)
