import datetime
import os
from functools import partial
from itertools import chain

from PIL import Image

from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.database_modules.database_logic import IncompleteDatabaseOperation, DBManager
from Logic.ProperLogic.database_modules.database_table_defs import Columns
from Logic.ProperLogic.handlers.handler_reset_cluster_ids import reset_cluster_ids
from Logic.ProperLogic.misc_helpers import log_error, overwrite_dict, starfilter, get_every_nth_item, \
    ignore_first_n_args_decorator
from Logic.ProperLogic.models_modules.models import Models
from Logic.ProperLogic.handlers.helpers import TO_TENSOR

# TODO: Cluster IDs should be reset right away!


def process_image_dir(cluster_dict, threshold=0.73, metric=2, **kwargs):
    """
    Extract faces from user-chosen images and cluster them

    :param threshold:
    :param metric:
    :param cluster_dict:
    :param kwargs:
    :return:
    """
    # TODO: Store entered paths(?) --> Makes it easier if user wants to revisit them, but probs rarely?
    images_path = user_choose_images_path()
    try:
        process_faces(images_path)
    except IncompleteDatabaseOperation:
        return

    cluster_dict_copy = cluster_dict.copy()

    def cluster_processed_faces(con):
        embeddings_with_ids = list(DBManager.get_all_embeddings(with_ids=True))

        # TODO: Call reclassify handler here?
        # TODO: Clear existing clusters? Issues with ids etc.????
        core_algorithm = CoreAlgorithm(metric=metric, classification_threshold=threshold)
        # passing result cluster dict already overwrites it
        clustering_result = core_algorithm.cluster_embeddings(embeddings_with_ids,
                                                              existing_clusters_dict=cluster_dict,
                                                              should_reset_cluster_ids=True,
                                                              final_clusters_only=False)
        _, modified_clusters_dict, removed_clusters_dict = clustering_result
        DBManager.overwrite_clusters_simplified(modified_clusters_dict, removed_clusters_dict, con=con,
                                                close_connections=False)
        reset_cluster_ids(con=con, close_connections=False)
        new_cluster_dict = DBManager.load_cluster_dict(con=con, close_connections=False)
        overwrite_dict(cluster_dict, new_cluster_dict)

    try:
        DBManager.connection_wrapper(cluster_processed_faces)
    except IncompleteDatabaseOperation:
        overwrite_dict(cluster_dict, cluster_dict_copy)


def process_faces(images_path, central_con=None, local_con=None, close_connections=True):
    if local_con is None:
        path_to_local_db = DBManager.get_local_db_file_path(images_path)
    else:
        path_to_local_db = None

    def process_faces_worker(central_con, local_con):
        DBManager.create_local_tables(drop_existing_tables=False, path_to_local_db=path_to_local_db, con=local_con,
                                      close_connections=False)
        extract_faces(images_path, central_con=central_con, local_con=local_con, close_connections=False)

    DBManager.connection_wrapper(process_faces_worker, path_to_local_db=path_to_local_db,
                                 central_con=central_con, local_con=local_con, with_central=True, with_local=True,
                                 close_connections=close_connections)


def extract_faces(path, check_if_known=True, central_con=None, local_con=None, close_connections=True):
    # TODO: Refactor (extract functions)? + rename
    # TODO: Generate Thumbnails differently? (E.g. via Image.thumbnail or sth. like that)
    # TODO: Store + update max_img_id and max_embedding_id somewhere rather than (always) get them via DB query?

    path_to_local_db = DBManager.get_local_db_file_path(path)
    path_id = DBManager.get_path_id(path)
    if path_id is None:
        # path not yet known
        path_id = DBManager.store_directory_path(path, con=central_con, close_connections=False)
        DBManager.store_path_id(path_id, path_to_local_db=path_to_local_db, con=local_con, close_connections=False)
    imgs_rel_paths_and_dates = set(DBManager.get_images_attributes(path_to_local_db=path_to_local_db))

    # Note: 'MAX' returns None / (None, ) as a default value
    max_img_id = DBManager.get_max_image_id(path_to_local_db=path_to_local_db)
    start_img_id = max_img_id + 1
    initial_max_embedding_id = DBManager.get_max_embedding_id()

    def get_counted_img_loader():
        img_loader = load_imgs_from_path(path, recursive=True, output_file_names=True, output_file_paths=True)
        return enumerate(img_loader, start=start_img_id)

    def store_embedding_row_dicts(con):
        # TODO: Also auto-increment emb_id etc.
        max_embedding_id = initial_max_embedding_id
        for img_id, (img_abs_path, img_rel_path, img) in get_counted_img_loader():
            # TODO: Implement automatic deletion cascade! (Using among other things on_conflict clause and FKs)
            #       ---> Done?
            # Check if image already stored --> don't process again
            # known = (name, last modified) as a pair known for this director
            last_modified = datetime.datetime.fromtimestamp(round(os.stat(img_abs_path).st_mtime))
            if check_if_known and (img_rel_path, last_modified) in imgs_rel_paths_and_dates:
                continue

            DBManager.store_image(img_id=img_id, rel_file_path=img_rel_path, last_modified=last_modified,
                                  path_to_local_db=path_to_local_db, con=local_con, close_connections=False)
            DBManager.store_image_path(img_id=img_id, path_id=path_id, con=central_con, close_connections=False)

            faces = Models.altered_mtcnn.forward_return_results(img)
            if not faces:
                log_error(f"no faces found in image '{img_abs_path}'")
                continue

            # TODO: Better way to create these row_dicts?
            embeddings_row_dicts = [{Columns.cluster_id.col_name: 'NULL',
                                     Columns.embedding.col_name: face_to_embedding(face),
                                     Columns.thumbnail.col_name: face,
                                     Columns.image_id.col_name: img_id,
                                     Columns.embedding_id.col_name: embedding_id}
                                    for embedding_id, face in enumerate(faces, start=max_embedding_id + 1)]
            DBManager.store_embeddings(embeddings_row_dicts, con=con, close_connections=False)
            max_embedding_id += len(faces)

    DBManager.connection_wrapper(store_embedding_row_dicts, con=central_con, close_connections=close_connections)


def user_choose_images_path():
    images_path = input('Please enter a path with images of people you would like to add.\n')
    while not os.path.exists(images_path):
        log_error(f"unable to find path '{images_path}'")
        print("\nPlease try again.")
        images_path = input('Please enter a path with images of people you would like to add.\n')

    # TODO: Implement check_if_known question(?)
    # check_if_known_decision = get_user_decision(
    #    "Should already processed images be processed again? This can be useful if for example some files have changed"
    #    " in a way the program doesn't recognize, or some faces from these images have been deleted and you would like"
    #    " to make them available again."
    # )
    # check_if_known = (check_if_known_decision == "n")
    return images_path


def faces_to_embeddings(faces):
    return map(face_to_embedding, faces)


def face_to_embedding(face):
    return Models.resnet(_to_tensor(face))


def _to_tensor(img):
    return TO_TENSOR(img).unsqueeze(0)


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
    for img_rel_path, img_abs_path in get_img_paths(dir_path, recursive, extensions, with_abs_paths=True):
        with Image.open(img_abs_path) as img:
            yield output_format_func(img_abs_path, img_rel_path, img)


def choose_args(indices, *args):
    # TODO: Use kwargs instead?
    # TODO: Use operator.itemgetter?
    return [arg for i, arg in enumerate(args) if i in indices]


def get_img_paths(base_dir_path, recursive=False, img_extensions=None, with_abs_paths=False):
    """
    Yield all image file paths in dir_path.
    """
    img_paths = _get_img_paths_worker(base_dir_path, recursive=recursive, img_extensions=img_extensions,
                                      with_abs_paths=with_abs_paths, rel_dir_path='')
    return img_paths


def _get_img_paths_worker(base_dir_path, recursive=False, img_extensions=None, with_abs_paths=False, rel_dir_path=''):
    @ignore_first_n_args_decorator(n=1)
    def is_img_known_extensions(obj_path):
        return is_img(obj_path, img_extensions)

    cur_dir = os.path.join(base_dir_path, rel_dir_path)
    # prepend the acquired directory tree of previous recursions
    object_rel_paths_in_dir = list(map(lambda obj: os.path.join(rel_dir_path, obj),
                                       os.listdir(cur_dir)))
    object_abs_paths_in_dir = list(map(partial(os.path.join, base_dir_path),
                                       object_rel_paths_in_dir))
    object_rel_and_abs_paths = zip(object_rel_paths_in_dir, object_abs_paths_in_dir)
    img_rel_and_abs_paths = starfilter(is_img_known_extensions, object_rel_and_abs_paths)
    img_rel_paths_in_dir = get_every_nth_item(img_rel_and_abs_paths, n=0)
    if not recursive:
        if with_abs_paths:
            return img_rel_and_abs_paths
        return img_rel_paths_in_dir

    def get_subdirs_img_paths(subdir_path):
        next_subdir = os.path.split(subdir_path)[-1]
        new_rel_dir_path = os.path.join(rel_dir_path, next_subdir)
        subdirs_img_paths = _get_img_paths_worker(base_dir_path, recursive=True, img_extensions=img_extensions,
                                                  with_abs_paths=with_abs_paths, rel_dir_path=new_rel_dir_path)
        return subdirs_img_paths

    subdir_abs_paths = filter(os.path.isdir, object_abs_paths_in_dir)
    subdirs_img_paths = map(get_subdirs_img_paths, subdir_abs_paths)
    img_objects_iterable = img_rel_and_abs_paths if with_abs_paths else img_rel_paths_in_dir
    all_img_objects = chain(img_objects_iterable, *subdirs_img_paths)
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
