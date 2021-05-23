from Logic.ProperLogic.database_modules.database_logic import DBManager, IncompleteDatabaseOperation


def reset_cluster_ids(cluster_dict=None, con=None, close_connections=True, **kwargs):
    # TODO: Create same function for embeddings ids once their removal is implemented?
    # TODO: Issue is clusters deleted -> cluster_id set to NULL in embeddings table -> reset doesn't work
    if cluster_dict is None:
        cluster_dict = DBManager.load_cluster_dict(con=con, close_connections=close_connections)
    old_ids, new_ids = cluster_dict.reset_ids()
    try:
        DBManager.reset_cluster_ids(old_ids, new_ids, cluster_dict, con=con, close_connections=close_connections)
    except IncompleteDatabaseOperation:
        cluster_dict.set_ids(new_ids, old_ids)
