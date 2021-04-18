from Logic.ProperLogic.cluster_modules.cluster_dict import ClusterDict
from Logic.ProperLogic.database_modules.database_logic import DBManager
from eval_handlers_versions.eval_process_image_dir import eval_process_image_dir

import f_measure


MAX_NUM_PROC_IMGS = 1
# IMAGES_PATH = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\Programming\BA\Logic\my_test\subset_cplfw\images'
IMAGES_PATH = r'..\my_test\subset_cplfw\images'
SAVE_RESULTS = True
SAVE_PATH = 'results'
DROP_TABLES = True


def main(images_path):
    if DROP_TABLES:
        drop_tables()
    cluster_dict = ClusterDict()
    emb_id_to_name_dict = eval_process_image_dir(cluster_dict, images_path, max_num_proc_imgs=MAX_NUM_PROC_IMGS)
    clusters = cluster_dict.get_clusters()
    f_measure.main(clusters, emb_id_to_name_dict, SAVE_RESULTS, SAVE_PATH)


def drop_tables():
    clear_central_tables()
    clear_local_tables()


def clear_local_tables():
    path_to_local_db_dir_path = IMAGES_PATH
    path_to_local_db = DBManager.get_db_path(path_to_local_db_dir_path, local=True)
    DBManager.clear_local_tables(path_to_local_db)


def clear_central_tables():
    DBManager.clear_central_tables()


if __name__ == '__main__':
    main(IMAGES_PATH)
