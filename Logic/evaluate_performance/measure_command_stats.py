import datetime
import os
import time

from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.handlers.handler_clear_data import clear_central_tables
from Logic.ProperLogic.handlers.handler_process_image_dir import load_imgs_from_path, face_to_embedding
from Logic.ProperLogic.handlers.handler_reclassify import reclassify
from Logic.ProperLogic.main_logic import init_program
from Logic.ProperLogic.models_modules.models import Models

from Logic.ProperLogic.database_modules.database_table_defs import Columns
from Logic.ProperLogic.database_modules.database_logic import DBManager, IncompleteDatabaseOperation
from Logic.ProperLogic.misc_helpers import log_error, overwrite_dict


EMBEDDINGS_PATH = 'Logic/ProperLogic/stored_embeddings'
CLUSTERS_PATH = 'stored_clusters'


def measure_commands():
    # TODO: process faces should be limited by n!
    write = False
    start, stop, step = 90, 450, 90
    COMMAND_STATS_PATH = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\BA Papers\Datasets\faces 1999 caltech\commands_stats.txt'
    DATASET_PATH = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\BA Papers\Datasets\faces 1999 caltech'

    def process_images_dir_measure(cluster_dict, n):
        images_path = DATASET_PATH
        try:
            print('------ PROCESSING FACES')
            process_faces_measure(images_path, n)
            print('------ DONE PROCESSING')
        except IncompleteDatabaseOperation as e:
            print('process_images_dir_measure error')
            log_error(e)
            return

        cluster_dict_copy = cluster_dict.copy()

        def cluster_processed_faces(con):
            embeddings_with_ids = list(DBManager.get_all_embeddings(with_ids=True))

            # TODO: Call reclassify handler here?
            # TODO: Clear existing clusters? Issues with ids etc.????
            core_algorithm = CoreAlgorithm()
            # passing result cluster dict already overwrites it
            clustering_result = core_algorithm.cluster_embeddings(embeddings_with_ids,
                                                                  existing_clusters_dict=cluster_dict,
                                                                  should_reset_cluster_ids=True,
                                                                  final_clusters_only=False)
            _, modified_clusters_dict, removed_clusters_dict = clustering_result
            DBManager.overwrite_clusters_simplified(modified_clusters_dict, removed_clusters_dict, con=con,
                                                    close_connections=False)
        try:
            DBManager.connection_wrapper(cluster_processed_faces)
        except IncompleteDatabaseOperation:
            overwrite_dict(cluster_dict, cluster_dict_copy)

    def process_faces_measure(images_path, n, central_con=None, local_con=None, close_connections=True):
        if local_con is None:
            path_to_local_db = DBManager.get_local_db_file_path(images_path)
        else:
            path_to_local_db = None

        def process_faces_worker(central_con, local_con):
            DBManager.create_local_tables(drop_existing_tables=False, path_to_local_db=path_to_local_db, con=local_con,
                                          close_connections=False)
            extract_faces_measure(images_path, n, central_con=central_con, local_con=local_con, close_connections=False)

        DBManager.connection_wrapper(process_faces_worker, path_to_local_db=path_to_local_db,
                                     central_con=central_con, local_con=local_con, with_central=True, with_local=True,
                                     close_connections=close_connections)

    def extract_faces_measure(path, n, check_if_known=True, central_con=None, local_con=None, close_connections=True):
        path_to_local_db = DBManager.get_local_db_file_path(path)
        path_id = DBManager.get_path_id(path)
        if path_id is None:
            # path not yet known
            path_id = DBManager.store_directory_path(path, con=central_con, close_connections=False)
            DBManager.store_path_id(path_id, path_to_local_db=path_to_local_db, con=local_con, close_connections=False)
        imgs_names_and_date = set(DBManager.get_images_attributes(path_to_local_db=path_to_local_db))

        # Note: 'MAX' returns None / (None, ) as a default value
        max_img_id = DBManager.get_max_image_id(path_to_local_db=path_to_local_db)
        start_img_id = max_img_id + 1
        initial_max_embedding_id = DBManager.get_max_embedding_id()

        def get_counted_img_loader():
            img_loader = load_imgs_from_path(path, recursive=True, output_file_names=True, output_file_paths=True)
            nums = range(start_img_id, start_img_id + n)
            return zip(nums, img_loader)

        def store_embedding_row_dicts(con):
            max_embedding_id = initial_max_embedding_id
            for img_id, (img_path, img_name, img) in get_counted_img_loader():
                # Check if image already stored --> don't process again
                # known = (name, last modified) as a pair known for this director
                last_modified = datetime.datetime.fromtimestamp(round(os.stat(img_path).st_mtime))
                if check_if_known and (img_name, last_modified) in imgs_names_and_date:
                    continue

                DBManager.store_image(img_id=img_id, file_name=img_name, last_modified=last_modified,
                                      path_to_local_db=path_to_local_db, con=local_con, close_connections=False)
                DBManager.store_image_path(img_id=img_id, path_id=path_id, con=central_con, close_connections=False)

                faces = Models.altered_mtcnn.forward_return_results(img)
                if not faces:
                    log_error(f"no faces found in image '{img_path}'")
                    continue

                embeddings_row_dicts = [{Columns.cluster_id.col_name: 'NULL',
                                         Columns.embedding.col_name: face_to_embedding(face),
                                         Columns.thumbnail.col_name: face,
                                         Columns.image_id.col_name: img_id,
                                         Columns.embedding_id.col_name: embedding_id}
                                        for embedding_id, face in enumerate(faces, start=max_embedding_id + 1)]
                DBManager.store_embeddings(embeddings_row_dicts, con=con, close_connections=False)
                max_embedding_id += len(faces)

        DBManager.connection_wrapper(store_embedding_row_dicts, con=central_con, close_connections=close_connections)

    def clear_data_measure(cluster_dict):
        local_db_dir_path = DATASET_PATH
        path_to_local_db = DBManager.get_local_db_file_path(local_db_dir_path)

        def clear_data_worker(central_con, local_con):
            DBManager.clear_local_tables(path_to_local_db, con=local_con, close_connections=False)
            clear_central_tables(con=central_con, close_connections=False)
            overwrite_dict(cluster_dict, dict())

        try:
            DBManager.connection_wrapper(clear_data_worker, path_to_local_db=path_to_local_db, with_central=True,
                                         with_local=True)
        except IncompleteDatabaseOperation as e:
            print('clear_data_measure error')
            log_error(e)

    cmds_list = [
        ('process_images_dir', process_images_dir_measure),
        ('reclassify', reclassify),
        ('clear_data', clear_data_measure),
    ]

    clear_data_measure(dict())
    commands = []
    for n in range(start, stop + 1, step):
        print(f'ITERATION: {n}')
        init_program()
        cluster_dict = DBManager.load_cluster_dict()
        for cmd_name, cmd in cmds_list:
            print(f'--- COMMAND: {cmd_name}')
            t1 = time.time()
            args = [cluster_dict] if cmd_name != 'process_images_dir' else [cluster_dict, n]
            cmd(*args)
            t2 = time.time()
            commands.append([cmd_name, n, t2 - t1])

    commands_str = '\n'.join(map(str, commands))
    if write:
        with open(COMMAND_STATS_PATH, 'w') as file:
            file.write(commands_str)


if __name__ == '__main__':
    measure_commands()
