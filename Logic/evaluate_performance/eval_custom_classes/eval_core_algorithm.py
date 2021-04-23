from Logic.ProperLogic.cluster_modules.cluster import Cluster
from Logic.ProperLogic.cluster_modules.cluster_dict import ClusterDict
from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.database_modules.database_logic import DBManager
from Logic.ProperLogic.misc_helpers import starfilterfalse

PRINT_PROGRESS = True
PROGRESS_STEPS = 100


class EvalCoreAlgorithm(CoreAlgorithm):
    def cluster_embeddings_no_split(self, embeddings, embeddings_ids=None, existing_clusters_dict=None,
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
        # embeddings = list(embeddings)
        if not embeddings:
            if final_clusters_only:
                return ClusterDict()
            return ClusterDict(), ClusterDict(), ClusterDict()

        if embeddings_ids is None:
            embeddings_with_ids = embeddings
        else:
            # if len(embeddings) > len(embeddings_ids):
            #     raise ValueError(f'Too few ids for embeddings ({len(embeddings_ids)} passed, but {len(embeddings)}'
            #                      f' needed)')
            embeddings_with_ids = zip(embeddings_ids, embeddings)

        if existing_clusters_dict is None:
            existing_clusters_dict = ClusterDict()
        else:
            # # Don't iterate over embeddings in existing clusters
            # embeddings_with_ids = dict(embeddings_with_ids)
            # existing_embeddings = existing_clusters_dict.get_embeddings()
            # remove_multiple(embeddings_with_ids, existing_embeddings)
            # embeddings_with_ids = embeddings_with_ids.items()
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
            print_progress(embedding_id, "embedding_id")
            closest_clusters = self.get_closest_clusters(cluster_dict, new_embedding)

            # find cluster containing the closest embedding to new_embedding
            shortest_emb_dist, closest_cluster = self.find_closest_cluster_to_embedding(closest_clusters, new_embedding)

            if shortest_emb_dist <= self.classification_threshold:
                closest_cluster.add_embedding(new_embedding, embedding_id)
                modified_clusters_ids.add(closest_cluster.cluster_id)
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


def print_progress(val, val_name):
    if PRINT_PROGRESS and val % PROGRESS_STEPS == 0:
        print(f'{val_name} -- {val}')
