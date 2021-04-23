import os
from itertools import product

import numpy as np

from Logic.ProperLogic.cluster_modules.cluster_dict import ClusterDict
from Logic.ProperLogic.main_logic import init_program
from Logic.evaluate_performance.eval_custom_classes.eval_dbmanager import EvalDBManager
from eval_handlers_versions.eval_process_image_dir import eval_process_image_dir

import f_measure

# TODO: Also store statistics about resulting clustering! (number of clusters, mean/median cluster size + variance)


# set to None to process all
MAX_NUM_PROC_IMGS = None
# IMAGES_PATH = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\Programming\BA\Logic\my_test\subset_cplfw\images'
IMAGES_PATH = r'..\my_test\subset_cplfw\images'
SAVE_RESULTS = True
SAVE_PATH = 'results_thresholds'
SAVE_FILE_NAME_POSTFIX = ''
DELETE_LOCAL_DB_FILE = False
DELETE_CENTRAL_DB_FILE = False
CLEAR_CLUSTERS = True

METRICS = [2]  # , 1.5, 1, 0.75, 0.5, 0.3, 0.2, 0.1, 0]
THRESHOLDS = np.linspace(0.5, 1.0, num=11)  # ==>  step size = 0.05
# METRICS = [20, 0]
# THRESHOLDS = [0.1, 10]


def run_evaluation(images_path):
    delete_db_files()
    init_program()

    if DELETE_CENTRAL_DB_FILE:
        cluster_dict = ClusterDict()
    else:
        cluster_dict = EvalDBManager.load_cluster_dict()
    eval_process_image_dir(cluster_dict, images_path, max_num_proc_imgs=MAX_NUM_PROC_IMGS)
    emb_id_to_name_dict = EvalDBManager.get_emb_id_to_name_dict(images_path=images_path)
    clusters = EvalDBManager.load_cluster_dict().get_clusters()
    f_measure.main(clusters, emb_id_to_name_dict, SAVE_RESULTS, SAVE_PATH, SAVE_FILE_NAME_POSTFIX)


def run_metric_evaluation(images_path):
    for counter, (threshold, metric) in enumerate(product(THRESHOLDS, METRICS), start=1):
        print('\n' f'-------------- STARTING EVAL {counter} --------------' '\n')
        delete_db_files()
        clear_clusters()
        init_program()

        if CLEAR_CLUSTERS:
            cluster_dict = ClusterDict()
        else:
            cluster_dict = EvalDBManager.load_cluster_dict()

        save_file_name_postfix = f'L{metric}__T{threshold}'.replace('.', '_point_')
        eval_process_image_dir(cluster_dict, images_path, max_num_proc_imgs=MAX_NUM_PROC_IMGS, metric=metric,
                               threshold=threshold)
        emb_id_to_name_dict = EvalDBManager.get_emb_id_to_name_dict(images_path=images_path)
        clusters = EvalDBManager.load_cluster_dict().get_clusters()
        f_measure.main(clusters, emb_id_to_name_dict, SAVE_RESULTS, SAVE_PATH, save_file_name_postfix)


def delete_db_files():
    if DELETE_CENTRAL_DB_FILE:
        delete_central_db_file()
    if DELETE_LOCAL_DB_FILE:
        delete_local_db_file()


def delete_central_db_file():
    path_to_central_db_file = EvalDBManager.get_central_db_file_path()
    os.remove(path_to_central_db_file)


def delete_local_db_file():
    path_to_local_db_file = EvalDBManager.get_local_db_file_path(IMAGES_PATH)
    os.remove(path_to_local_db_file)


def clear_clusters():
    if CLEAR_CLUSTERS:
        EvalDBManager.clear_clusters()


if __name__ == '__main__':
    run_metric_evaluation(IMAGES_PATH)
