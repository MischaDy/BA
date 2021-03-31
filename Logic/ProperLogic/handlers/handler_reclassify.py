from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.database_modules.database_logic import DBManager
from Logic.ProperLogic.misc_helpers import overwrite_list


def reclassify(clusters, **kwargs):
    embeddings_with_ids = DBManager.get_all_embeddings(with_ids=True)
    existing_clusters = DBManager.get_certain_clusters()
    reclassified_clusters = CoreAlgorithm.cluster_embeddings(embeddings=embeddings_with_ids,
                                                             existing_clusters=existing_clusters)
    overwrite_list(clusters, reclassified_clusters)
