from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.database_modules.database_logic import DBManager
from Logic.ProperLogic.handlers.handler_reset_cluster_ids import reset_cluster_ids
from Logic.ProperLogic.misc_helpers import overwrite_list, log_error


def reclassify(clusters, **kwargs):
    embeddings_with_ids = list(DBManager.get_all_embeddings(with_ids=True))
    if not embeddings_with_ids:
        log_error('no embeddings found, nothing to edit')
        return
    certain_clusters = DBManager.get_certain_clusters()
    reclassified_clusters = CoreAlgorithm.cluster_embeddings(embeddings=embeddings_with_ids,
                                                             existing_clusters=certain_clusters,
                                                             final_clusters_only=True)

    def overwrite_clusters(con):
        DBManager.remove_clusters(remove_all=True, con=con, close_connections=False)
        reset_cluster_ids(reclassified_clusters)
        DBManager.store_clusters(reclassified_clusters, con=con, close_connections=False)
        overwrite_list(clusters, reclassified_clusters)

    DBManager.connection_wrapper(overwrite_clusters)
