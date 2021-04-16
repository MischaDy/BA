from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.database_modules.database_logic import DBManager, IncompleteDatabaseOperation
from Logic.ProperLogic.misc_helpers import log_error, overwrite_dict


def reclassify(cluster_dict, **kwargs):
    def reclassify_worker(con):
        embeddings_with_ids = list(DBManager.get_all_embeddings(with_ids=True))
        if not embeddings_with_ids:
            log_error('no embeddings found, nothing to edit')
            return
        new_cluster_dict = DBManager.get_certain_clusters()
        clustering_result = CoreAlgorithm.cluster_embeddings(embeddings=embeddings_with_ids,
                                                             existing_clusters_dict=new_cluster_dict,
                                                             final_clusters_only=True)
        new_cluster_dict.reset_ids()
        _, modified_clusters_dict, removed_clusters_dict = clustering_result
        DBManager.overwrite_clusters(new_cluster_dict, modified_clusters_dict, removed_clusters_dict, con=con,
                                     close_connections=False)
        DBManager.store_clusters()
        overwrite_dict(cluster_dict, new_cluster_dict)

    try:
        DBManager.connection_wrapper(reclassify_worker)
    except IncompleteDatabaseOperation:
        pass
