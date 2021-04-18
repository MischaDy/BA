from Logic.ProperLogic.handlers.handler_process_image_dir import *


PRINT_PROGRESS = True
PROGRESS_STEPS = 10


def eval_process_image_dir(cluster_dict, images_path, max_num_proc_imgs=None):
    try:
        faces_rows, emb_id_to_name_dict = eval_get_faces_rows(images_path, max_num_proc_imgs=max_num_proc_imgs)
    except IncompleteDatabaseOperation:
        return

    cluster_dict_copy = cluster_dict.copy()

    def eval_process_image_dir_worker(con):
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
                                     emb_id_to_img_id_dict=emb_id_to_img_id_dict, con=con,
                                     close_connections=False)

    try:
        DBManager.connection_wrapper(eval_process_image_dir_worker)
        return emb_id_to_name_dict
    except IncompleteDatabaseOperation:
        overwrite_dict(cluster_dict, cluster_dict_copy)


def eval_get_faces_rows(images_path, max_num_proc_imgs=None, central_con=None, local_con=None, close_connections=True):
    if local_con is None:
        path_to_local_db = DBManager.get_local_db_file_path(images_path)
    else:
        path_to_local_db = None

    def eval_get_faces_rows_worker(central_con, local_con):
        DBManager.create_local_tables(drop_existing_tables=False, path_to_local_db=path_to_local_db, con=local_con,
                                      close_connections=False)
        faces_rows, emb_id_to_name_dict = eval_extract_faces(images_path, max_num_proc_imgs=max_num_proc_imgs,
                                                             central_con=central_con,
                                                             local_con=local_con, close_connections=False)
        return list(faces_rows), emb_id_to_name_dict

    faces_rows, emb_id_to_name_dict = DBManager.connection_wrapper(eval_get_faces_rows_worker,
                                                                   path_to_local_db=path_to_local_db,
                                                                   central_con=central_con, local_con=local_con,
                                                                   with_central=True,
                                                                   with_local=True, close_connections=close_connections)
    return faces_rows, emb_id_to_name_dict


def print_progress(val, val_name):
    if PRINT_PROGRESS and val % PROGRESS_STEPS == 0:
        print(f'{val_name} -- {val}')


def eval_extract_faces(path, check_if_known=True, max_num_proc_imgs=None, central_con=None, local_con=None,
                       close_connections=True):
    path_to_local_db = DBManager.get_local_db_file_path(path)
    img_loader = load_imgs_from_path(path, recursive=True, output_file_names=True, output_file_paths=True)

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
        emb_id_to_name_dict = {}
        img_id = max_img_id + 1
        max_embedding_id = initial_max_embedding_id

        if max_num_proc_imgs is not None:
            counted_img_loader = zip(range(1, max_num_proc_imgs + 1), img_loader)
        else:
            counted_img_loader = enumerate(img_loader, start=1)

        for counter, (img_path, img_name, img) in counted_img_loader:
            print_progress(counter, 'image')

            last_modified = datetime.datetime.fromtimestamp(round(os.stat(img_path).st_mtime))
            if check_if_known and (img_name, last_modified) in imgs_names_and_date:
                continue

            DBManager.store_image(img_id=img_id, file_name=img_name, last_modified=last_modified,
                                  path_to_local_db=path_to_local_db, con=local_con, close_connections=False)
            DBManager.store_image_path(img_id=img_id, path_id=path_id, con=central_con, close_connections=False)

            img_faces = cut_out_faces(Models.mtcnn, img)
            if len(img_faces) != 1:
                error_msg = f'more than 1 face encountered!' '\n' f'{counter}, {img_path}, {img_name}'
                raise RuntimeError(error_msg)

            cur_faces_rows = []
            for embedding_id, face in enumerate(img_faces, start=max_embedding_id + 1):
                cur_faces_rows.append({Columns.thumbnail.col_name: face,
                                       Columns.image_id.col_name: img_id,
                                       Columns.embedding_id.col_name: embedding_id})
                emb_id_to_name_dict[embedding_id] = img_name
            faces_rows.extend(cur_faces_rows)
            max_embedding_id += len(img_faces)
            img_id += 1

        return faces_rows, emb_id_to_name_dict

    faces_rows, emb_id_to_name_dict = DBManager.connection_wrapper(extract_faces_worker, central_con=central_con,
                                                                   local_con=local_con,
                                                                   with_central=True, with_local=True,
                                                                   close_connections=close_connections)
    return faces_rows, emb_id_to_name_dict
