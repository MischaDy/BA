import datetime
import os
import sqlite3
import weakref
from functools import partial

import torchvision
from PIL import Image
from facenet_pytorch.models.utils.detect_face import get_size, crop_resize

from cluster import Cluster
from core_algorithm import CoreAlgorithm
from database_logic import DBManager
from database_table_defs import Tables, Columns
from models import Models
from misc_helpers import log_error, clean_str, wait_for_any_input, get_every_nth_item, have_equal_type_names, \
    overwrite_list, get_user_decision

IMG_PATH = 'Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive'

# TODO: Where to put these?
TO_PIL_IMAGE = torchvision.transforms.ToPILImage()
TO_TENSOR = torchvision.transforms.ToTensor()


# INPUT_SIZE = [112, 112]


# TODO: Make handlers class
# TODO: Split this file?

class Command:
    # TODO: add 'help' command
    terminating_tokens = ('halt', 'stop', 'quit', 'exit',)
    commands = {}

    def __init__(self, cmd_name, cmd_desc, handler=None, handler_params=None):
        if handler_params is None:
            handler_params = []
        self.cmd_name = cmd_name
        self.cmd_desc = cmd_desc
        self.handler = handler
        self.handler_params = handler_params
        type(self).commands[self.cmd_name] = self

    def __eq__(self, other):
        # TODO: Implement more strict checking?
        if not have_equal_type_names(self, other):
            return False
        return self.cmd_name == other.cmd_name

    def get_cmd_name(self):
        return self.cmd_name

    def set_cmd_name(self, cmd_name):
        self.cmd_name = cmd_name

    def get_cmd_description(self):
        return self.cmd_desc

    def set_cmd_description(self, new_cmd_desc):
        self.cmd_desc = new_cmd_desc

    def get_handler(self):
        return self.handler

    def set_handler(self, new_handler):
        self.handler = new_handler

    def get_handler_params(self):
        return self.handler_params

    def set_handler_params(self, new_handler_params):
        self.handler_params = new_handler_params

    @classmethod
    def get_commands(cls):
        return cls.commands

    @classmethod
    def get_command_names(cls):
        return cls.commands.keys()

    @classmethod
    def get_command_descriptions(cls):
        return [(cmd.cmd_name, cmd.cmd_desc) for cmd in cls.commands.values()]

    @classmethod
    def remove_command(cls, cmd_name):
        try:
            cls.commands.pop(cmd_name)
        except KeyError:
            log_error(f"Could not remove unknown command '{cmd_name}'")

    @classmethod
    def get_command(cls, cmd_name):
        try:
            cmd = cls.commands[cmd_name]
        except KeyError:
            log_error(f"Could not remove unknown command '{cmd_name}'")
            return None
        return cmd


class Commands:
    process_imgs = Command('process images', 'select new faces')
    edit_faces = Command('edit faces', 'edit existing faces')
    find = Command('find person', 'find person')
    reclassify = Command('reclassify', 'reclassify individuals')
    show_cluster = Command('show cluster', 'show a cluster')
    label_clusters = Command('label clusters', '(re-)name clusters')

    @classmethod
    def initialize(cls):
        cls.process_imgs.set_handler(handler_process_image_dir)
        cls.edit_faces.set_handler(handler_edit_faces)
        cls.find.set_handler(handler_find_person)
        cls.reclassify.set_handler(handler_reclassify)
        cls.show_cluster.set_handler(handler_show_cluster)
        cls.label_clusters.set_handler(handler_label_clusters)


# ----- COMMAND PROCESSING -----

def handler_label_clusters(**kwargs):
    pass


def handler_edit_faces(clusters, **kwargs):
    # TODO: Finish implementing
    # TODO: Include option to delete people (and remember that in case same dir is read again? --> Probs optional)

    # TODO: Make sure user-selected labels are treated correctly in clustering!
    if not clusters:
        log_error('no clusters found, nothing to edit')
        return

    continue_cluster = ''
    while not continue_cluster.startswith('n'):
        cluster = user_choose_cluster(clusters)

        continue_face = ''
        while not continue_face.startswith('n'):
            face = user_choose_face(cluster)
            new_label = user_choose_face_label(cluster)
            set_cluster_label(cluster, new_label, clusters)
            continue_face = clean_str(input('Relabel another face in this cluster?\n'))
        continue_cluster = clean_str(input('Choose another cluster?\n'))


def handler_find_person(**kwargs):
    # TODO: Needed? Is this substantially different from show_cluster? (Probs only regarding multiple clusters with same
    #       label...)
    # TODO: Implement
    pass


def handler_reclassify(**kwargs):
    # TODO: Implement
    pass


def handler_show_cluster(clusters_path, **kwargs):
    # TODO: Finish implementing
    should_continue = ''
    while not should_continue.startswith('n'):
        cluster_name, cluster_path = user_choose_cluster(clusters_path)
        _output_cluster_content(cluster_name, cluster_path)
        ...
        should_continue = clean_str(input('Choose another cluster?\n'))


def handler_process_image_dir(db_manager: DBManager, clusters, **kwargs):
    # TODO: Refactor + improve efficiency
    # TODO: Store entered paths(?) --> Makes it easier if user wants to revisit them, but probs rarely?
    # Extract faces from user-chosen images and cluster them
    faces_rows = list(user_choose_imgs(db_manager))
    if not faces_rows:
        return
    # TODO: Implement correct processing of faces_rows!
    # {'thumbnail': <PIL.Image.Image image mode=RGB size=160x160 at 0x21C0B1BAC48>, 'image_id': 1, 'embedding_id': 1}

    # TODO: Extract this dictionary-querying as function?

    embedding_ids = list(map(lambda row_dict: row_dict[Columns.embedding_id.col_name],
                             faces_rows))
    thumbnails = map(lambda row_dict: row_dict[Columns.thumbnail.col_name],
                     faces_rows)
    faces = map(lambda row_dict: row_dict[Columns.thumbnail.col_name],
                faces_rows)
    image_ids = map(lambda row_dict: row_dict[Columns.image_id.col_name],
                    faces_rows)
    embeddings = list(faces_to_embeddings(faces))
    clustering_result = CoreAlgorithm.cluster_embeddings(embeddings, embedding_ids, db_manager, existing_clusters=clusters)
    updated_clusters, modified_clusters, removed_clusters = clustering_result

    emb_id_to_face_dict = dict(zip(embedding_ids, thumbnails))
    emb_id_to_img_id_dict = dict(zip(embedding_ids, image_ids))
    db_manager.remove_clusters(list(removed_clusters))
    db_manager.store_clusters(list(modified_clusters), emb_id_to_face_dict, emb_id_to_img_id_dict)
    overwrite_list(clusters, updated_clusters)


def faces_to_embeddings(faces):
    return map(lambda face: Models.resnet(_to_tensor(face)), faces)


def user_choose_imgs(db_manager):
    # TODO: Refactor! (too many different tasks, function name non-descriptive)
    # TODO: Make user choose path
    # TODO: (Permanently) disable dropping of existing tables
    images_path = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\Programming\BA\Logic\my_test\facenet_Test\group_imgs'  # user_choose_path()
    path_to_local_db = db_manager.get_db_path(images_path, local=True)
    db_manager.create_tables(create_local=True,
                             path_to_local_db=path_to_local_db,
                             drop_existing_tables=False)
    faces_rows = extract_faces(images_path, db_manager)
    return faces_rows


def user_choose_path():
    path = input('Please enter a path with images of people you would like to add.\n')
    while not os.path.exists(path):
        log_error(f"Unable to find path '{path}'")
        print("\nPlease try again.")
        path = input('Please enter a path with images of people you would like to add.\n')
    return path  # IMG_PATH


def extract_faces(path, db_manager: DBManager, check_if_known=True):
    # TODO: Refactor (extract functions)? + rename
    # TODO: Generate Thumbnails differently? (E.g. via Image.thumbnail or sth. like that)
    # TODO: Store + update max_img_id and max_embedding_id somewhere rather than (always) get them via DB query?
    # TODO: Outsource db interactions to input-output logic?
    # TODO: Check if max_embedding_id maths checks out!

    path_to_local_db = db_manager.get_db_path(path, local=True)

    imgs_names_and_date = set(db_manager.get_imgs_attrs(path_to_local_db=path_to_local_db))

    # Note: 'MAX' returns None / (None, ) as a default value
    max_img_id = db_manager.get_max_image_id(path_to_local_db=path_to_local_db)
    max_embedding_id = db_manager.get_max_embedding_id()

    faces_rows = []
    img_loader = load_imgs_from_path(path, output_file_names=True, output_file_paths=True)
    img_id = max_img_id + 1
    for img_path, img_name, img in img_loader:
        # TODO: Implement automatic deletion cascade! (Using among other things on_conflict clause and FKs)
        # Check if image already stored --> Don't process again!
        # known = name and last modified as a pair known for this directory
        last_modified = datetime.datetime.fromtimestamp(round(os.stat(img_path).st_mtime))
        if check_if_known and (img_name, last_modified) in imgs_names_and_date:
            continue

        img_faces = cut_out_faces(Models.mtcnn, img)
        img_row = {Columns.image_id.col_name: img_id,
                   Columns.file_name.col_name: img_name,
                   Columns.last_modified.col_name: last_modified}
        db_manager.store_in_table(Tables.images_table, [img_row], path_to_local_db=path_to_local_db)
        cur_faces_rows = [{Columns.thumbnail.col_name: face,
                           Columns.image_id.col_name: img_id,
                           Columns.embedding_id.col_name: embedding_id}
                          for embedding_id, face in enumerate(img_faces, start=max_embedding_id+1)]
        faces_rows.extend(cur_faces_rows)
        max_embedding_id += len(img_faces)
        img_id += 1

    return faces_rows


def cut_out_faces(mtcnn, img):
    """
    NOTE: This part is copied from the extract_face function in facenet_pytorch/models/utils/detect_face.py,
    since this particular functionality is only provided for saving, not returning the face pictures.
    """
    # TODO: Use a file buffer or something like that to save from the original function instead of doing this??
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
    # TODO: Finish implementing (what's missing?)
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


def _to_tensor(img):
    return TO_TENSOR(img).unsqueeze(0)


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


def _output_cluster_content(cluster_name, cluster_path):
    wait_for_any_input(f'Which face image in the cluster "{cluster_name}" would you like to view?')
    # TODO: finish
    # TODO: output faces and (-> separate function?) allow choice of image


def set_cluster_label(cluster, new_label, clusters):
    # TODO: Implement
    pass


# --- i/o helpers ---

def user_choose_cluster(clusters):
    # TODO: Refactor
    cluster_ids = list(clusters.get_cluster_ids())
    print_clusters(clusters)
    chosen_cluster_id = get_user_input_of_type(class_=int, obj_name='cluster id')
    while chosen_cluster_id not in cluster_ids:
        log_error(f'cluster "{chosen_cluster_id}" not found; Please try again.')
        print_clusters(clusters)
        chosen_cluster_id = get_user_input_of_type(class_=int, obj_name='cluster id')

    chosen_cluster = clusters.get_cluster_by_id(chosen_cluster_id)
    return chosen_cluster


def get_user_input_of_type(class_, obj_name):
    user_input = None
    while not isinstance(user_input, class_):
        try:
            user_input = class_(input())
        except ValueError:
            log_error(f'{obj_name} must be of type {class_}, not {type(obj_name)}. Please try again.')
    return user_input


def user_choose_face(cluster):
    # TODO: Finish implementing
    # TODO: Refactor
    # TODO: WIP!
    face_ids = cluster.get_face_ids()
    faces = ...  # How to
    print_faces(cluster)
    chosen_face_id = input()
    while chosen_face_id not in face_ids:
        log_error(f'face "{chosen_face_id}" not found; Please try again.')
    print_faces(cluster)
    chosen_face_id = input()

    chosen_cluster = clusters.get_cluster_by_id(chosen_face_id)
    return chosen_cluster


def user_choose_face_label(cluster):
    # TODO: Implement
    pass


def print_clusters(clusters):
    # TODO: print limited number of clusters at a time (Enter=continue)
    cluster_labels = clusters.get_cluster_labels()
    cluster_ids = clusters.get_cluster_ids()
    clusters_strs = (f"- Cluster {cluster_id} ('{label}')"
                     for cluster_id, label in zip(cluster_ids, cluster_labels))
    wait_for_any_input('\nPlease type the id of the cluster you would like to view. (Press any key to continue.)')
    print('\n'.join(clusters_strs))


def print_faces(faces):
    # TODO: Finish implementing
    # TODO: How to print/show faces???
    # TODO: print limited number of faces at a time (Enter=continue)
    cluster_labels = clusters.get_cluster_labels()
    cluster_ids = clusters.get_cluster_ids()
    clusters_str = map(lambda cluster_id, label: f"- Cluster {cluster_id} ('{label}')",
                       zip(cluster_ids, cluster_labels))
    wait_for_any_input('Please type the id of the cluster you would like to view. (Press any key to continue.)')
    print('\n'.join(clusters_str))


# ----- FILE I/O -----

def get_clusters_gen(clusters_path, return_names=True):
    file_obj_names = os.listdir(clusters_path)
    file_obj_paths = map(lambda obj_name: os.path.join(clusters_path, obj_name), file_obj_names)
    clusters_names_and_paths = filter(lambda obj_tup: os.path.isdir(obj_tup[1]), zip(file_obj_names, file_obj_paths))
    if return_names:
        return clusters_names_and_paths
    # return only paths
    return get_every_nth_item(clusters_names_and_paths, n=1)


# ----- MISC -----

handlers = {
    'processimgs': handler_process_image_dir,
    'showcluster': handler_show_cluster,
}
