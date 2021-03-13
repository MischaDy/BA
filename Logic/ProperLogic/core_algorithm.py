from functools import partial

from Logic.ProperLogic.database_logic import DBManager
from Logic.ProperLogic.misc_helpers import log_error, remove_items
from input_output_logic import *

from itertools import count

from timeit import default_timer
import logging
logging.basicConfig(level=logging.INFO)


# TODO: Add proper comments!

# TODO: Select good params


# design decision re. splitting: check for cluster size first, because greater efficiency (short-circuit), although if
# both cluster-size and too far from center apply, it would be better to start reclustering with new embedding.


CLUSTERS_PATH = 'stored_clusters'
EMBEDDINGS_PATH = 'stored_embeddings'


class CoreAlgorithm:
    path_to_central_db = os.path.join(DBManager.db_files_path, DBManager.central_db_file_name)
    path_to_local_db = os.path.join(DBManager.db_files_path, DBManager.local_db_file_name)

    # 0.53  # OR 0.73 cf. Bijl - A comparison of clustering algorithms for face clustering
    classification_threshold = 0.95
    # TODO: Is that a sensible value?
    reclustering_threshold = 2 * classification_threshold  # max dist to cluster center before an emb. gets own cluster

    max_cluster_size = 100
    max_num_total_comps = 1000
    max_num_cluster_comps = max_num_total_comps // max_cluster_size  # maximum number of clusters to compute distance to

    # TODO: remove
    num_embeddings_to_classify = 100

    @classmethod
    def cluster_embeddings(cls, embeddings, embedding_ids, existing_clusters=None):
        """
        Build clusters from face embeddings stored in the given path using the specified classification threshold.
        (Currently handled as: All embeddings closer than the distance given by the classification threshold are placed
        in the same cluster. If cluster_save_path is set, store the resulting clusters as directories in the given path.

        :param embeddings: Either a path (string) to the directory wherein embeddings are located or an iterable
        containing the embeddings
        :param embedding_ids: Ordered iterable with the embedding ids. Must be at least as long as embeddings.
        :param existing_clusters:
        :return:
        """
        # TODO: What should the output be? Should this be an in-place method??
        # TODO: Improve efficiency?
        # TODO: Allow embeddings_ids to be none? Get next id via DB query?
        # TODO: Allow embeddings_ids to be shorter than embeddings and 'fill up' remaining ids?
        if len(embeddings) > len(embedding_ids):
            raise ValueError(f'Too few ids for embeddings ({len(embedding_ids)} passed, but {len(embeddings)} needed)')

        if existing_clusters is None:
            existing_clusters = []
        embeddings_with_ids = zip(embedding_ids, embeddings)
        clusters = existing_clusters

        # iterate over remaining embeddings
        logging.info('START iteration over embeddings')
        time1 = default_timer()
        counter_vals = range(2, cls.num_embeddings_to_classify + 1) if cls.num_embeddings_to_classify >= 0 else count(2)
        counter = 0
        for counter, (embedding_id, new_embedding) in zip(counter_vals, embeddings_with_ids):
            if counter % 100 == 0:
                logging.info(f' --- Current embedding number: {counter}')

            # sort clusters by distance from their center to embedding; only consider closest clusters
            clusters_by_center_dist = sorted(clusters,
                                             key=lambda cluster: cluster.compute_dist_to_center(new_embedding))
            closest_clusters = clusters_by_center_dist[:cls.max_num_cluster_comps]

            # find cluster containing the closest embedding to new_embedding
            shortest_emb_dist, closest_cluster = cls.find_closest_cluster_to_embedding(closest_clusters, new_embedding)

            if shortest_emb_dist <= cls.classification_threshold:
                closest_cluster.add_embedding(new_embedding, embedding_id)
                distant_embeddings = cls.get_embs_too_far_from_center(closest_cluster)
                if len(distant_embeddings) > 0 or cls.is_cluster_too_big(closest_cluster):
                    cls.split_cluster(closest_cluster, clusters, start_embeddings=distant_embeddings)
            else:
                clusters.append(Cluster([new_embedding], [embedding_id]))

        logging.info(f' --- Last embedding number: {counter}')
        logging.info(f'END iteration over embeddings')
        logging.info(f'Time spent on embeddings: {default_timer() - time1}')

        return clusters

    @classmethod
    def is_cluster_too_big(cls, cluster):
        return cls.max_cluster_size is not None and cluster.get_size() >= cls.max_cluster_size

    @classmethod
    def get_embs_too_far_from_center(cls, cluster):
        if cls.reclustering_threshold is None:
            return []
        return list(filter(lambda emb: cluster.compute_dist_to_center(emb) > cls.reclustering_threshold,
                           cluster.get_embeddings()))

    @classmethod
    def find_closest_cluster_to_embedding(cls, clusters, embedding, return_dist=True):
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
    def split_cluster(cls, closest_cluster, clusters, start_embeddings=None):
        """
        Recluster given cluster without further (indirectly recursive) splitting.

        :param closest_cluster: ...
        :param clusters: Iterable of currently assigned clusters
        :param start_embeddings: Embeddings which to cluster first
        """
        # TODO: Compare with first embedding or with cluster center?
        # TODO: Improve efficiency!
        embeddings = list(closest_cluster.get_embeddings())
        if start_embeddings is not None:
            remove_items(embeddings, start_embeddings)
            embeddings = start_embeddings + embeddings

        cluster_start_emb1, cluster_start_emb2 = cls.find_most_distant_embeddings(embeddings)
        remove_items(embeddings, [cluster_start_emb1, cluster_start_emb2])
        label = closest_cluster.label
        new_cluster1, new_cluster2 = Cluster(cluster_start_emb1, label=label), Cluster(cluster_start_emb2, label=label)
        for embedding in embeddings:
            dist_to_emb1, dist_to_emb2 = map(embedding.dist, [cluster_start_emb1, cluster_start_emb2])
            closer_cluster = new_cluster1 if dist_to_emb1 < dist_to_emb2 else new_cluster2
            closer_cluster.add_embedding(embedding)

        clusters.extend([new_cluster1, new_cluster2])
        clusters.remove(closest_cluster)

    @staticmethod
    def find_most_distant_embeddings(embeddings):
        """
        Return the two embeddings which have the greatest distance between them.
        :param embeddings:
        :return:
        """
        # TODO: Improve efficiency
        max_dist = -1
        max_dist_embs = (None, None)
        for ind1, emb1 in enumerate(embeddings):
            for emb2 in embeddings[ind1+1:]:
                cur_dist = Cluster.compute_dist(emb1, emb2)
                if cur_dist > max_dist:
                    max_dist = cur_dist
                    max_dist_embs = (emb1, emb2)
        return max_dist_embs

