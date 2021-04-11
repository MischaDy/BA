from Logic.ProperLogic.database_modules.database_logic import DBManager, IncompleteDatabaseOperation


def reset_cluster_ids(cluster_dict, **kwargs):
    # TODO: Create same function for embeddings ids once their removal is implemented?
    old_ids, new_ids = clusters.reset_ids()
    try:
        DBManager.reset_cluster_ids(old_ids, new_ids)
    except IncompleteDatabaseOperation:
        clusters.set_ids(new_ids, old_ids)
