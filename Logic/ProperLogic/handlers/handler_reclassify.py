from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.database_modules.database_logic import DBManager
from Logic.ProperLogic.handlers.handler_reset_cluster_ids import reset_cluster_ids
from Logic.ProperLogic.misc_helpers import overwrite_list, log_error, overwrite_dict


def reclassify(cluster_dict, **kwargs):
    # TODO: Do embeddings_with_ids within certain clusters need to be removed? Done automatically???

    embeddings_with_ids = list(DBManager.get_all_embeddings(with_ids=True))
    if not embeddings_with_ids:
        log_error('no embeddings found, nothing to edit')
        return
    certain_clusters_dict = DBManager.get_certain_clusters()
    new_cluster_dict = CoreAlgorithm.cluster_embeddings(embeddings=embeddings_with_ids,
                                                        existing_clusters_dict=certain_clusters_dict,
                                                        final_clusters_only=True)

    def overwrite_clusters(con):
        DBManager.remove_clusters(remove_all=True, con=con, close_connections=False)
        reset_cluster_ids(new_cluster_dict)
        DBManager.store_clusters(new_cluster_dict, con=con, close_connections=False)
        overwrite_dict(cluster_dict, new_cluster_dict)

    # TODO: How to handle possible exception here?
    #       --> Can prevent reset_cluster_ids call if removing clusters first? Does that work?!
    DBManager.connection_wrapper(overwrite_clusters)
