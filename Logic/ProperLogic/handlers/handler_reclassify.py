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
        core_algorithm = CoreAlgorithm()
        clustering_result = core_algorithm.cluster_embeddings(embeddings=local_embeddings_with_ids,
                                                              existing_clusters_dict=new_cluster_dict,
                                                              should_reset_cluster_ids=True,
                                                              final_clusters_only=False)
        _, modified_clusters_dict, removed_clusters_dict = clustering_result
        DBManager.overwrite_clusters(new_cluster_dict, removed_clusters_dict, no_new_embs=True,
                                     clear_clusters=True, con=con, close_connections=False)
        overwrite_dict(cluster_dict, new_cluster_dict)

    try:
        DBManager.connection_wrapper(reclassify_worker, con=con, close_connections=close_connections)
    except IncompleteDatabaseOperation:
        pass
