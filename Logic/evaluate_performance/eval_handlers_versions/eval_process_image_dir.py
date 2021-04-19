from Logic.ProperLogic.handlers.handler_process_image_dir import *


PRINT_PROGRESS = True
PROGRESS_STEPS = 10


def eval_process_image_dir(cluster_dict, images_path, max_num_proc_imgs=None):
    Models.altered_mtcnn.keep_all = False
    try:
        emb_id_to_name_dict = eval_get_emb_id_to_name_dict(images_path, max_num_proc_imgs=max_num_proc_imgs)
    except IncompleteDatabaseOperation:
        return

    cluster_dict_copy = cluster_dict.copy()

    def eval_process_image_dir_worker(con):
        embeddings_with_ids = list(DBManager.get_all_embeddings(with_ids=True))
        if not embeddings_with_ids:
            return

        # passing result cluster dict already overwrites it
        clustering_result = CoreAlgorithm.cluster_embeddings(embeddings_with_ids,
                                                             existing_clusters_dict=cluster_dict,
                                                             should_reset_cluster_ids=True,
                                                             final_clusters_only=False)
        _, modified_clusters_dict, removed_clusters_dict = clustering_result
        DBManager.overwrite_clusters_simplified(modified_clusters_dict, removed_clusters_dict, con=con,
                                                close_connections=False)

    try:
        DBManager.connection_wrapper(eval_process_image_dir_worker)
        return emb_id_to_name_dict
    except IncompleteDatabaseOperation:
        overwrite_dict(cluster_dict, cluster_dict_copy)


def eval_get_emb_id_to_name_dict(images_path, max_num_proc_imgs=None, central_con=None, local_con=None, close_connections=True):
    if local_con is None:
        path_to_local_db = DBManager.get_local_db_file_path(images_path)
    else:
        path_to_local_db = None

    def eval_get_faces_rows_worker(central_con, local_con):
        DBManager.create_local_tables(drop_existing_tables=False, path_to_local_db=path_to_local_db, con=local_con,
                                      close_connections=False)
        emb_id_to_name_dict = eval_extract_faces(images_path, max_num_proc_imgs=max_num_proc_imgs,
                                                 central_con=central_con,
                                                 local_con=local_con, close_connections=False)
        return emb_id_to_name_dict

    emb_id_to_name_dict = DBManager.connection_wrapper(eval_get_faces_rows_worker,
                                                       path_to_local_db=path_to_local_db,
                                                       central_con=central_con, local_con=local_con,
                                                       with_central=True,
                                                       with_local=True, close_connections=close_connections)
    return emb_id_to_name_dict


def print_progress(val, val_name):
    if PRINT_PROGRESS and val % PROGRESS_STEPS == 0:
        print(f'{val_name} -- {val}')


def eval_extract_faces(path, check_if_known=True, max_num_proc_imgs=None, central_con=None, local_con=None,
                       close_connections=True):
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
        if max_num_proc_imgs is not None:
            return zip(range(start_img_id, max_num_proc_imgs + 1), img_loader)
        return enumerate(img_loader, start=start_img_id)

    def get_emb_id_to_name_dict(central_con, local_con):
        print('----- get_emb_id_to_name_dict -----')
        emb_id_to_name_dict = {}
        # TODO: Also auto-increment emb_id etc.
        embedding_id = initial_max_embedding_id + 1
        for img_id, (img_path, img_name, img) in get_counted_img_loader():
            print_progress(img_id, 'image')

            last_modified = datetime.datetime.fromtimestamp(round(os.stat(img_path).st_mtime))
            if check_if_known and (img_name, last_modified) in imgs_names_and_date:
                continue

            DBManager.store_image(img_id=img_id, file_name=img_name, last_modified=last_modified,
                                  path_to_local_db=path_to_local_db, con=local_con, close_connections=False)
            DBManager.store_image_path(img_id=img_id, path_id=path_id, con=central_con, close_connections=False)

            emb_id_to_name_dict[embedding_id] = img_name

        return emb_id_to_name_dict

    def store_embedding_row_dicts(central_con):
        print('----- get_embedding_row_dicts -----')
        # TODO: Also auto-increment emb_id etc.
        embedding_id = initial_max_embedding_id + 1
        for img_id, (img_path, img_name, img) in get_counted_img_loader():
            print_progress(img_id, 'image')

            last_modified = datetime.datetime.fromtimestamp(round(os.stat(img_path).st_mtime))
            if check_if_known and (img_name, last_modified) in imgs_names_and_date:
                continue

            face = Models.altered_mtcnn.forward_return_results(img)
            if face is None:
                log_error(f"no faces found in image '{img_path}'")
                continue

            embedding_row_dict = {Columns.cluster_id.col_name: 'NULL',
                                  Columns.embedding.col_name: face_to_embedding(face),
                                  Columns.thumbnail.col_name: face,
                                  Columns.image_id.col_name: img_id,
                                  Columns.embedding_id.col_name: embedding_id}
            DBManager.store_embedding(embedding_row_dict, con=central_con, close_connections=False)
            embedding_id += 1

    def eval_extract_faces_worker(central_con, local_con):
        emb_id_to_name_dict = get_emb_id_to_name_dict(central_con, local_con)
        store_embedding_row_dicts(central_con)
        return emb_id_to_name_dict

    emb_id_to_name_dict = DBManager.connection_wrapper(eval_extract_faces_worker, central_con=central_con,
                                                       local_con=local_con,
                                                       with_central=True, with_local=True,
                                                       close_connections=close_connections)
    return emb_id_to_name_dict
