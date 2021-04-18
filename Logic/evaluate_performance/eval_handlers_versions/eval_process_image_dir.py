from Logic.ProperLogic.handlers.handler_process_image_dir import *


PRINT_PROGRESS = True
PROGRESS_STEPS = 10


def eval_process_image_dir(cluster_dict, images_path):
    try:
        faces_rows = eval_get_faces_rows(images_path)
    except IncompleteDatabaseOperation:
        return

    cluster_dict_copy = cluster_dict.copy()

    def eval_process_image_dir_worker(central_con):
        if not faces_rows:
            return

        embeddings_ids = list(map(lambda row_dict: row_dict[Columns.embedding_id.col_name],
                                  faces_rows))
        thumbnails = map(lambda row_dict: row_dict[Columns.thumbnail.col_name],
                         faces_rows)
        image_ids = map(lambda row_dict: row_dict[Columns.image_id.col_name],
                        faces_rows)
        faces = map(lambda row_dict: row_dict[Columns.thumbnail.col_name],
                    faces_rows)

        embeddings = list(faces_to_embeddings(faces))

        emb_id_to_face_dict = dict(zip(embeddings_ids, thumbnails))
        emb_id_to_img_id_dict = dict(zip(embeddings_ids, image_ids))

        # passing result cluster dict already overwrites it
        clustering_result = CoreAlgorithm.cluster_embeddings(embeddings, embeddings_ids,
                                                             existing_clusters_dict=cluster_dict,
                                                             should_reset_cluster_ids=True,
                                                             final_clusters_only=False)
        _, modified_clusters_dict, removed_clusters_dict = clustering_result
        DBManager.overwrite_clusters(modified_clusters_dict, removed_clusters_dict,
                                     emb_id_to_face_dict=emb_id_to_face_dict,
                                     emb_id_to_img_id_dict=emb_id_to_img_id_dict, con=central_con,
                                     close_connections=False)

    try:
        DBManager.connection_wrapper(eval_process_image_dir_worker)
    except IncompleteDatabaseOperation:
        overwrite_dict(cluster_dict, cluster_dict_copy)


def eval_get_faces_rows(images_path, central_con=None, local_con=None, close_connections=True):
    path_to_local_db = DBManager.get_db_path(images_path, local=True)

    def eval_get_faces_rows_worker(central_con, local_con):
        faces_rows = list(eval_get_images(images_path, path_to_local_db=path_to_local_db,
                                          central_con=central_con, local_con=local_con,
                                          close_connections=False))
        return faces_rows

    faces_rows = DBManager.connection_wrapper(eval_get_faces_rows_worker, path_to_local_db=path_to_local_db,
                                              central_con=central_con, local_con=local_con, with_central=True,
                                              with_local=True, close_connections=close_connections)
    return faces_rows


def eval_get_images(images_path, path_to_local_db=None, central_con=None, local_con=None,
                    close_connections=True):
    if path_to_local_db is None and local_con is None:
        path_to_local_db = DBManager.get_db_path(images_path, local=True)

    def eval_get_images_worker(central_con, local_con):
        DBManager.create_local_tables(drop_existing_tables=False, path_to_local_db=path_to_local_db, con=local_con,
                                      close_connections=False)
        faces_rows = eval_extract_faces(images_path, central_con=central_con, local_con=local_con,
                                        close_connections=False)
        return faces_rows

    faces_rows = DBManager.connection_wrapper(eval_get_images_worker, path_to_local_db=path_to_local_db,
                                              central_con=central_con, local_con=local_con, with_central=True,
                                              with_local=True, close_connections=close_connections)
    return faces_rows


def print_progress(val, val_name):
    if PRINT_PROGRESS and val % PROGRESS_STEPS == 0:
        print(f'{val_name} -- {val}')


def eval_extract_faces(path, check_if_known=True, central_con=None, local_con=None, close_connections=True):
    path_to_local_db = DBManager.get_db_path(path, local=True)
    img_loader = load_imgs_from_path(path, output_file_names=True, output_file_paths=True)

    def extract_faces_worker(central_con, local_con):
        imgs_names_and_date = set(DBManager.get_images_attributes(path_to_local_db=path_to_local_db))

        # Note: 'MAX' returns None / (None, ) as a default value
        max_img_id = DBManager.get_max_image_id(path_to_local_db=path_to_local_db)
        initial_max_embedding_id = DBManager.get_max_embedding_id()

        path_id = DBManager.get_path_id(path)
        if path_id is None:
            # path not yet known
            path_id = DBManager.store_directory_path(path, con=central_con, close_connections=False)
            DBManager.store_path_id(path_id, path_to_local_db=path_to_local_db, con=local_con, close_connections=False)

        faces_rows = []
        img_id = max_img_id + 1
        max_embedding_id = initial_max_embedding_id
        for counter, (img_path, img_name, img) in enumerate(img_loader, start=1):
            print_progress(counter, 'image')

            last_modified = datetime.datetime.fromtimestamp(round(os.stat(img_path).st_mtime))
            if check_if_known and (img_name, last_modified) in imgs_names_and_date:
                continue

            DBManager.store_image(img_id=img_id, file_name=img_name, last_modified=last_modified,
                                  path_to_local_db=path_to_local_db, con=local_con, close_connections=False)
            DBManager.store_image_path(img_id=img_id, path_id=path_id, con=central_con, close_connections=False)

            img_faces = cut_out_faces(Models.mtcnn, img)
            cur_faces_rows = [{Columns.thumbnail.col_name: face,
                               Columns.image_id.col_name: img_id,
                               Columns.embedding_id.col_name: embedding_id}
                              for embedding_id, face in enumerate(img_faces, start=max_embedding_id + 1)]
            faces_rows.extend(cur_faces_rows)
            max_embedding_id += len(img_faces)
            img_id += 1

        return faces_rows

    faces_rows = DBManager.connection_wrapper(extract_faces_worker, central_con=central_con, local_con=local_con,
                                              with_central=True, with_local=True, close_connections=close_connections)
    return faces_rows
