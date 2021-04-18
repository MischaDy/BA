import os

from Logic.ProperLogic.cluster_modules.cluster_dict import ClusterDict
from Logic.ProperLogic.database_modules.database_logic import DBManager
from Logic.ProperLogic.main_logic import init_program
from eval_handlers_versions.eval_process_image_dir import eval_process_image_dir

import f_measure


# set to None to process all
MAX_NUM_PROC_IMGS = 10
# IMAGES_PATH = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\Programming\BA\Logic\my_test\subset_cplfw\images'
IMAGES_PATH = r'..\my_test\subset_cplfw\images'
SAVE_RESULTS = True
SAVE_PATH = 'results'
DROP_TABLES = True


# TODO: Fix <class 'sqlite3.IntegrityError'>, ('UNIQUE constraint failed: images.image_id',)

def run_evaluation(images_path):
    if DROP_TABLES:
        delete_db_files()
    init_program()

    cluster_dict = ClusterDict()
    emb_id_to_name_dict = eval_process_image_dir(cluster_dict, images_path, max_num_proc_imgs=MAX_NUM_PROC_IMGS)
    clusters = cluster_dict.get_clusters()
    f_measure.main(clusters, emb_id_to_name_dict, SAVE_RESULTS, SAVE_PATH)


def delete_db_files():
    delete_central_db_file()
    delete_local_db_file()


def delete_central_db_file():
    path_to_central_db_file = DBManager.get_central_db_file_path()
    os.remove(path_to_central_db_file)


def delete_local_db_file():
    path_to_local_db_file = DBManager.get_local_db_file_path(IMAGES_PATH)
    os.remove(path_to_local_db_file)


if __name__ == '__main__':
    run_evaluation(IMAGES_PATH)
