import datetime
import os
from functools import partial

from PIL import Image
from facenet_pytorch.models.utils.detect_face import get_size, crop_resize

from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.database_modules.database_logic import IncompleteDatabaseOperation, DBManager
from Logic.ProperLogic.database_modules.database_table_defs import Columns
from Logic.ProperLogic.misc_helpers import log_error, overwrite_dict
from Logic.ProperLogic.models import Models
from Logic.ProperLogic.handlers.helpers import TO_TENSOR


def process_image_dir(cluster_dict, **kwargs):
    """
    Extract faces from user-chosen images and cluster them

    :param cluster_dict:
    :param kwargs:
    :return:
    """
    # TODO: Refactor + improve efficiency
    # TODO: Store entered paths(?) --> Makes it easier if user wants to revisit them, but probs rarely?

    def process_image_dir_worker(con):
        faces_rows = list(user_choose_images())
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

        clustering_result = CoreAlgorithm.cluster_embeddings(embeddings, embeddings_ids,
                                                             existing_clusters_dict=cluster_dict,
                                                             final_clusters_only=False)
        updated_clusters_dict, modified_clusters_dict, removed_clusters_dict = clustering_result
        DBManager.remove_clusters(removed_clusters_dict, con=con, close_connections=False)
        DBManager.store_clusters(modified_clusters_dict, emb_id_to_face_dict, emb_id_to_img_id_dict, con=con,
                                 close_connections=False)
        overwrite_dict(cluster_dict, updated_clusters_dict)

    try:
        DBManager.connection_wrapper(process_image_dir_worker)
    except IncompleteDatabaseOperation:
        pass


def user_choose_images(global_con=None, local_con=None, close_connections=True):
    # TODO: Refactor! (too many different tasks, function name non-descriptive)
    # TODO: make user user choose path
    images_path = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\Programming\BA\Logic\my_test\facenet_Test\group_imgs'
    path_to_local_db = DBManager.get_db_path(images_path, local=True)

    def user_choose_images_worker(global_con, local_con):
        DBManager.create_local_tables(drop_existing_tables=False, path_to_local_db=path_to_local_db,
                                      con=local_con, close_connections=False)
        faces_rows = extract_faces(images_path, global_con=global_con, local_con=local_con, close_connections=False)
        return faces_rows

    faces_rows = DBManager.connection_wrapper(user_choose_images_worker, path_to_local_db=path_to_local_db,
                                              global_con=global_con, local_con=local_con,
                                              close_connections=close_connections)

    # TODO: Implement check_if_known question(?)
    # check_if_known_decision = get_user_decision(
    #    "Should already processed images be processed again? This can be useful if for example some files have changed"
    #    " in a way the program doesn't recognize, or some faces from these images have been deleted and you would like"
    #    " to make them available again."
    # )
    # check_if_known = (check_if_known_decision == "n")

    return faces_rows


def user_choose_images_path():
    path = input('Please enter a path with images of people you would like to add.\n')
    while not os.path.exists(path):
        log_error(f"unable to find path '{path}'")
        print("\nPlease try again.")
        path = input('Please enter a path with images of people you would like to add.\n')
    return path  # IMG_PATH


def faces_to_embeddings(faces):
    return map(lambda face: Models.resnet(_to_tensor(face)), faces)


def _to_tensor(img):
    return TO_TENSOR(img).unsqueeze(0)


def extract_faces(path, check_if_known=True, global_con=None, local_con=None, close_connections=True):
    # TODO: Refactor (extract functions)? + rename
    # TODO: Generate Thumbnails differently? (E.g. via Image.thumbnail or sth. like that)
    # TODO: Store + update max_img_id and max_embedding_id somewhere rather than (always) get them via DB query?

    path_to_local_db = DBManager.get_db_path(path, local=True)
    img_loader = load_imgs_from_path(path, output_file_names=True, output_file_paths=True)

    def extract_faces_worker(global_con, local_con):
        # TODO: Outsource as function to DBManager?
        # TODO: Check whether known locally and centrally separately?

        imgs_names_and_date = set(DBManager.get_images_attributes(path_to_local_db=path_to_local_db))

        # Note: 'MAX' returns None / (None, ) as a default value
        max_img_id = DBManager.get_max_image_id(path_to_local_db=path_to_local_db)
        initial_max_embedding_id = DBManager.get_max_embedding_id()

        path_id = DBManager.get_path_id(path)
        if path_id is None:
            # path not yet known
            path_id = DBManager.store_directory_path(path, con=global_con, close_connections=False)
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
            DBManager.store_image_path(img_id=img_id, path_id=path_id, con=global_con, close_connections=False)

            img_faces = cut_out_faces(Models.mtcnn, img)
            # TODO: Better way to create these row_dicts?
            cur_faces_rows = [{Columns.thumbnail.col_name: face,
                               Columns.image_id.col_name: img_id,
                               Columns.embedding_id.col_name: embedding_id}
                              for embedding_id, face in enumerate(img_faces, start=max_embedding_id + 1)]
            faces_rows.extend(cur_faces_rows)
            max_embedding_id += len(img_faces)
            img_id += 1

        return faces_rows

    if global_con is None:
        global_con = DBManager.open_central_connection()
    if local_con is None:
        local_con = DBManager.open_local_connection(path_to_local_db)

    faces_rows = DBManager.connection_wrapper(extract_faces_worker, global_con=global_con, local_con=local_con,
                                              close_connections=close_connections)
    return faces_rows


def load_imgs_from_path(dir_path, output_file_names=False, output_file_paths=False, extensions=None):
    """
    Yield all images in the given directory.
    If img_img_extensions is empty, all files are assumed to be images. Otherwise, only files with extensions appearing
    in the set will be returned.

    :param output_file_names: Whether the tensor should be yielded together with the corresponding file name
    :param output_file_paths: Whether the tensor should be yielded together with the corresponding file path
    :param dir_path: Directory containing images
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
    for img_name in get_img_names(dir_path, extensions):
        img_path = os.path.join(dir_path, img_name)
        with Image.open(img_path) as img:
            yield output_format_func(img_path, img_name, img)


def choose_args(indices, *args):
    # TODO: Use kwargs instead?
    # TODO: Use operator.itemgetter?
    return [arg for i, arg in enumerate(args) if i in indices]


def cut_out_faces(mtcnn, img):
    """
    NOTE: This part is copied from the extract_face function in facenet_pytorch/models/utils/detect_face.py,
    since this particular functionality is only provided for saving, not for returning the face pictures.
    """
    # TODO: Use a file buffer or something like that to save from the original function instead of doing this??
    #       --> Not possible, it expects file path
    boxes, _ = mtcnn.detect(img)
    image_size, mtcnn_margin = mtcnn.image_size, mtcnn.margin
    faces = []
    for box in boxes:
        margin = [
            mtcnn_margin * (box[2] - box[0]) / (image_size - mtcnn_margin),
            mtcnn_margin * (box[3] - box[1]) / (image_size - mtcnn_margin),
            ]
        raw_image_size = get_size(img)
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_image_size[0])),
            int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]

        face = crop_resize(img, box, image_size)
        faces.append(face)
    return faces


def get_img_names(dir_path, img_extensions=None):
    """
    Yield all image file paths in dir_path.
    """

    # TODO: Implement recursive option?
    # TODO: Put function outside?
    def is_img_known_extensions(obj_name):
        return is_img(os.path.join(dir_path, obj_name), img_extensions)

    image_paths = filter(is_img_known_extensions, os.listdir(dir_path))
    return image_paths


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
