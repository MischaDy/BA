import os
from itertools import product

from Logic.ProperLogic.cluster_modules.cluster_dict import ClusterDict
from Logic.ProperLogic.database_modules.database_logic import IncompleteDatabaseOperation
from Logic.ProperLogic.handlers.handler_clear_data import clear_clustering
from Logic.ProperLogic.main_logic import init_program
from Logic.evaluate_performance.eval_custom_classes.eval_dbmanager import EvalDBManager
from Logic.evaluate_performance.eval_handlers_versions.eval_process_image_dir import eval_process_image_dir

from Logic.evaluate_performance import f_measure

# TODO: Also store statistics about resulting clustering! (number of clusters, mean/median cluster size + variance)


# set to None to process all
MAX_NUM_PROC_IMGS = None
SAVE_RESULTS = True
SAVE_FILE_NAME_POSTFIX = ''


# def run_evaluation(images_path):
#     delete_db_files()
#     init_program()
#
#     if DELETE_CENTRAL_DB_FILE:
#         cluster_dict = ClusterDict()
#     else:
#         cluster_dict = EvalDBManager.load_cluster_dict()
#     eval_process_image_dir(cluster_dict, images_path, max_num_proc_imgs=MAX_NUM_PROC_IMGS)
#     emb_id_to_img_name_dict = EvalDBManager.get_emb_id_to_name_dict(images_path=images_path)
#     clusters = EvalDBManager.load_cluster_dict().get_clusters()
#     f_measure.main(clusters, emb_id_to_img_name_dict, SAVE_RESULTS, SAVE_PATH, SAVE_FILE_NAME_POSTFIX)


def run_metric_evaluation(images_path, are_same_person_func, save_path, thresholds=(0.73,), metrics=(2,),
                          delete_central_db_file=False, delete_local_db_file=False, clear_clusters=True):
    for counter, (threshold, metric) in enumerate(product(thresholds, metrics), start=1):
        print('\n' f'-------------- STARTING EVAL {counter} --------------' '\n')
        try:
            clear_clustering()
        except IncompleteDatabaseOperation:
            input('Issue!')
        delete_db_files(delete_central_db_file, delete_local_db_file, images_path)
        init_program()

        if clear_clusters:
            cluster_dict = ClusterDict()
        else:
            cluster_dict = EvalDBManager.load_cluster_dict()

        save_file_name_postfix = f'L{metric}__T{threshold}'.replace('.', '_point_')
        eval_process_image_dir(cluster_dict, images_path, max_num_proc_imgs=MAX_NUM_PROC_IMGS, metric=metric,
                               threshold=threshold)
        emb_id_to_name_dict = EvalDBManager.get_emb_id_to_name_dict(images_path=images_path)
        clusters = EvalDBManager.load_cluster_dict().get_clusters()
        f_measure.main(clusters, emb_id_to_name_dict, are_same_person_func, SAVE_RESULTS,
                       save_path, save_file_name_postfix)


def delete_db_files(should_delete_central_db_file, should_delete_local_db_file, images_path):
    if should_delete_central_db_file:
        delete_central_db_file()
    if should_delete_local_db_file:
        delete_local_db_file(images_path)


def delete_db_file(path_to_db_file, is_local):
    confirm = input('--- WARNING ---\n' f"Really delete {'local' if is_local else 'central'} db files")
    if not confirm:
        exit()

    try:
        os.remove(path_to_db_file)
    except FileNotFoundError:
        pass


def delete_central_db_file():
    path_to_central_db_file = EvalDBManager.get_central_db_file_path()
    delete_db_file(path_to_central_db_file)


def delete_local_db_file(images_path):
    path_to_local_db_file = EvalDBManager.get_local_db_file_path(images_path)
    delete_db_file(path_to_local_db_file)


# if __name__ == '__main__':
#     run_metric_evaluation(IMAGES_PATH)
