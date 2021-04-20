import datetime
import os
from functools import partial
from itertools import chain

from PIL import Image

from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.database_modules.database_logic import IncompleteDatabaseOperation, DBManager
from Logic.ProperLogic.database_modules.database_table_defs import Columns
from Logic.ProperLogic.misc_helpers import log_error, overwrite_dict, starfilter, get_every_nth_item, \
    ignore_first_n_args_decorator
from Logic.ProperLogic.models_modules.models import Models
from Logic.ProperLogic.handlers.helpers import TO_TENSOR


# TODO: Transfer code from eval_process_handler to here!


def process_image_dir(cluster_dict, **kwargs):
    """
    Extract faces from user-chosen images and cluster them

    :param cluster_dict:
    :param kwargs:
    :return:
    """
    # TODO: Split embeddings storage and clustering, so one can be complete even if the other fails
    # TODO: Refactor + improve efficiency
    # TODO: Store entered paths(?) --> Makes it easier if user wants to revisit them, but probs rarely?

    images_path = user_choose_images_path()
    try:
        path_to_local_db = DBManager.get_local_db_file_path(images_path)
    except IncompleteDatabaseOperation:
        return

    cluster_dict_copy = cluster_dict.copy()

    def process_image_dir_worker(central_con, local_con):
        faces_rows = list(user_choose_images(images_path, path_to_local_db=path_to_local_db, central_con=central_con,
                                             local_con=local_con, close_connections=False))
        if not faces_rows:
            return

        # TODO: Extract this dictionary-querying as function?
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

        # TODO: Call reclassify handler here?
        # TODO: Clear existing clusters? Issues with ids etc.????
        # passing result cluster dict already overwrites it
        core_algorithm = CoreAlgorithm()
        clustering_result = core_algorithm.cluster_embeddings(embeddings, embeddings_ids,
                                                              existing_clusters_dict=cluster_dict,
                                                              should_reset_cluster_ids=True,
                                                              final_clusters_only=False)
        _, modified_clusters_dict, removed_clusters_dict = clustering_result
        DBManager.overwrite_clusters(modified_clusters_dict, removed_clusters_dict,
                                     emb_id_to_face_dict=emb_id_to_face_dict,
                                     emb_id_to_img_id_dict=emb_id_to_img_id_dict, con=central_con,
                                     close_connections=False)

    try:
        DBManager.connection_wrapper(process_image_dir_worker, path_to_local_db=path_to_local_db, with_central=True,
                                     with_local=True)
    except IncompleteDatabaseOperation:
        overwrite_dict(cluster_dict, cluster_dict_copy)


def user_choose_images(images_path, path_to_local_db=None, central_con=None, local_con=None, close_connections=True):
    # TODO: Are the following lines a good default?
    if path_to_local_db is None and local_con is None:
        path_to_local_db = DBManager.get_local_db_file_path(images_path)

    def user_choose_images_worker(central_con, local_con):
        DBManager.create_local_tables(drop_existing_tables=False, path_to_local_db=path_to_local_db,
                                      con=local_con, close_connections=False)
        faces_rows = extract_faces(images_path, central_con=central_con, local_con=local_con, close_connections=False)
        return faces_rows

    faces_rows = DBManager.connection_wrapper(user_choose_images_worker, path_to_local_db=path_to_local_db,
                                              central_con=central_con, local_con=local_con, with_central=True,
                                              with_local=True, close_connections=close_connections)

    # TODO: Implement check_if_known question(?)
    # check_if_known_decision = get_user_decision(
    #    "Should already processed images be processed again? This can be useful if for example some files have changed"
    #    " in a way the program doesn't recognize, or some faces from these images have been deleted and you would like"
    #    " to make them available again."
    # )
    # check_if_known = (check_if_known_decision == "n")

    return faces_rows


def user_choose_images_path():
    # TODO: make user user choose path
    # images_path = input('Please enter a path with images of people you would like to add.\n')
    images_path = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\Programming\BA\Logic\my_test\facenet_Test\group_imgs'
    while not os.path.exists(images_path):
        log_error(f"unable to find path '{images_path}'")
        print("\nPlease try again.")
        images_path = input('Please enter a path with images of people you would like to add.\n')
    return images_path


def faces_to_embeddings(faces):
    return map(face_to_embedding, faces)


def face_to_embedding(face):
    return Models.resnet(_to_tensor(face))


def _to_tensor(img):
    return TO_TENSOR(img).unsqueeze(0)


def extract_faces(path, check_if_known=True, central_con=None, local_con=None, close_connections=True):
    # TODO: Refactor (extract functions)? + rename
    # TODO: Generate Thumbnails differently? (E.g. via Image.thumbnail or sth. like that)
    # TODO: Store + update max_img_id and max_embedding_id somewhere rather than (always) get them via DB query?

    path_to_local_db = DBManager.get_local_db_file_path(path)
    img_loader = load_imgs_from_path(path, output_file_names=True, output_file_paths=True)

    def extract_faces_worker(central_con, local_con):
        # TODO: Outsource as function to DBManager?
        # TODO: Check whether known locally and centrally separately?

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
        for img_path, img_name, img in img_loader:
            # TODO: Implement automatic deletion cascade! (Using among other things on_conflict clause and FKs)
            #       ---> Done?
            # Check if image already stored --> don't process again
            # known = (name, last modified) as a pair known for this directory
            last_modified = datetime.datetime.fromtimestamp(round(os.stat(img_path).st_mtime))
            if check_if_known and (img_name, last_modified) in imgs_names_and_date:
                continue

            DBManager.store_image(img_id=img_id, file_name=img_name, last_modified=last_modified,
                                  path_to_local_db=path_to_local_db, con=local_con, close_connections=False)
            DBManager.store_image_path(img_id=img_id, path_id=path_id, con=central_con, close_connections=False)
            # img_faces = cut_out_faces(Models.mtcnn, img)
            img_faces = Models.altered_mtcnn.forward_return_results(img)
            # TODO: Better way to create these row_dicts?
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


def load_imgs_from_path(dir_path, recursive=False, output_file_names=False, output_file_paths=False, extensions=None):
    """
    Yield all images in the given directory.
    If img_img_extensions is empty, all files are assumed to be images. Otherwise, only files with extensions appearing
    in the set will be returned.

    :param dir_path: Directory containing images
    :param recursive: Whether subdirectories should also be processed (recursively)
    :param output_file_names: Whether the tensor should be yielded together with the corresponding file name
    :param output_file_paths: Whether the tensor should be yielded together with the corresponding file path
    :param extensions: Iterable of file extensions considered images, e.g. ['jpg', 'png']. Default: 'jpg' and 'png'.
    filtering
    :return: Yield(!) tuples of image_names and PIL images contained in this folder
    """
    # TODO: More pythonic way to select function based on condition??
    indices = []
    if output_file_paths:
        indices.append(0)
    if output_file_names:
        indices.append(1)
    indices.append(2)
    output_format_func = partial(choose_args, indices)
    for img_name, img_path in get_img_names(dir_path, recursive, extensions, with_paths=True):
        with Image.open(img_path) as img:
            yield output_format_func(img_path, img_name, img)


def choose_args(indices, *args):
    # TODO: Use kwargs instead?
    # TODO: Use operator.itemgetter?
    return [arg for i, arg in enumerate(args) if i in indices]


def get_img_names(dir_path, recursive=False, img_extensions=None, with_paths=False):
    """
    Yield all image file paths in dir_path.
    """
    # TODO: Put the following function outside?
    @ignore_first_n_args_decorator(n=1)
    def is_img_known_extensions(obj_path):
        return is_img(obj_path, img_extensions)

    object_names_in_dir = list(os.listdir(dir_path))
    object_paths_in_dir = list(map(partial(os.path.join, dir_path),
                                   object_names_in_dir))
    object_names_and_paths = zip(object_names_in_dir, object_paths_in_dir)
    img_names_and_paths = starfilter(is_img_known_extensions, object_names_and_paths)
    img_names_in_dir = get_every_nth_item(img_names_and_paths, n=0)
    if not recursive:
        if with_paths:
            return img_names_and_paths
        return img_names_in_dir

    dir_paths = filter(os.path.isdir, object_paths_in_dir)
    subdirs_img_names_iterables = map(partial(get_img_names, recursive=True, with_paths=with_paths),
                                      dir_paths)
    img_objects_iterable = img_names_and_paths if with_paths else img_names_in_dir
    all_img_objects = chain(img_objects_iterable, *subdirs_img_names_iterables)
    return all_img_objects


def is_img(obj_path, img_extensions=None):
    """

    :param obj_path: Path to an object
    :param img_extensions: Iterable of extensions. Default: 'jpg', 'jpeg' and 'png'.
    :return: Whether obj_path ends with
    """
    if img_extensions is None:
        img_extensions = {'jpg', 'jpeg', 'png'}
    else:
        img_extensions = set(map(lambda s: s.lower(), img_extensions))
    if not os.path.isfile(obj_path):
        return False
    return any(map(lambda ext: obj_path.endswith(ext), img_extensions))


class FaceDetectionError(RuntimeError):
    pass
