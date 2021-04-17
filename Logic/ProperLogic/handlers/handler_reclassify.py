from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.database_modules.database_logic import DBManager, IncompleteDatabaseOperation
from Logic.ProperLogic.misc_helpers import log_error, overwrite_dict


# TODO: Fix bug where after reclassification, clusters seem to contain no images.
#       Is bc stored cluster ids in embeddings table arent reset and clusters aren't properly removed/overwritten
#

def reclassify(cluster_dict, embeddings_with_ids=None, con=None, close_connections=True, **kwargs):
    def reclassify_worker(con):
        # all operations in worker, so if any DB operation raises error, it is caught
        if embeddings_with_ids is not None:
            local_embeddings_with_ids = embeddings_with_ids
        else:
            local_embeddings_with_ids = list(DBManager.get_all_embeddings(with_ids=True))

        if not local_embeddings_with_ids:
            log_error('no embeddings found, nothing to edit')
            return

        new_cluster_dict = DBManager.get_certain_clusters()
        clustering_result = CoreAlgorithm.cluster_embeddings(embeddings=local_embeddings_with_ids,
                                                             existing_clusters_dict=new_cluster_dict,
                                                             final_clusters_only=False)
        _, modified_clusters_dict, removed_clusters_dict = clustering_result
        DBManager.overwrite_clusters(modified_clusters_dict, removed_clusters_dict, no_new_embs=True, con=con,
                                     close_connections=False)
        new_cluster_dict.reset_ids()
        overwrite_dict(cluster_dict, new_cluster_dict)

    try:
        DBManager.connection_wrapper(reclassify_worker, con=con, close_connections=close_connections)
    except IncompleteDatabaseOperation:
        pass
