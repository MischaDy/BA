import torchvision
from facenet_pytorch.models.utils.detect_face import get_size, crop_resize

from Logic.ProperLogic.core_algorithm import CoreAlgorithm
from Logic.ProperLogic.database_logic import *
from models import Models
from misc_helpers import log_error, clean_str, wait_for_any_input, get_every_nth_item, have_equal_type_names, \
    split_items

IMG_PATH = 'Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive'

TO_PIL_IMAGE = torchvision.transforms.ToPILImage()
TO_TENSOR = torchvision.transforms.ToTensor()

# INPUT_SIZE = [112, 112]


# TODO: Make enum-style Commands class like for ColumnSchema etc.?

class Command:
    # TODO: add 'help' command
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
    # TODO: Make enum?
    add = Command('add', 'select new faces')
    edit = Command('edit', 'edit existing faces')
    find = Command('find', 'find individual')
    reclassify = Command('reclassify', 'reclassify individuals')
    show_cluster = Command('showcluster', 'show a cluster')


def initialize_commands():
    # TODO: Finish!
    # TODO: Add params?
    Commands.add.set_handler(handler_add_new_embeddings)
    Commands.edit.set_handler(handler_edit_faces)
    Commands.find.set_handler(handler_find_person)
    Commands.reclassify.set_handler(handler_reclassify)
    Commands.show_cluster.set_handler(handler_show_cluster)


# ----- COMMAND PROCESSING -----

def handler_edit_faces(**kwargs):
    # TODO: Implement
    pass


def handler_find_person(**kwargs):
    # TODO: Implement
    pass


def handler_reclassify(**kwargs):
    # TODO: Implement
    pass


def handler_show_cluster(clusters_path, **kwargs):
    # TODO: Finish implementation
    should_continue = ''
    while 'n' not in should_continue:
        cluster_name, cluster_path = _user_choose_cluster(clusters_path)
        _output_cluster_content(cluster_name, cluster_path)
        ...
        should_continue = clean_str(input('Choose another cluster?\n'))


def handler_add_new_embeddings(db_manager, clusters, **kwargs):
    # TODO: Finish implementing
    # TODO: Improve efficiency? (only one loop for row creation)
    # TODO: How to know, what to store in DB?? ---> Improve efficiency! Make sure, DB handles uniqueness correctly!
    # Extract faces from user-chosen images and cluster them
    face_ids, faces = split_items(list(user_choose_imgs(db_manager)))
    embeddings = list(faces_to_embeddings(faces))
    CoreAlgorithm.cluster_embeddings(embeddings, face_ids, existing_clusters=clusters)

    # Store clusters in cluster_attributes table of DB
    attributes_row_dicts = [
        {
            Columns.cluster_id_col.col_name: cluster.cluster_id,
            Columns.label_col.col_name: cluster.label,
            Columns.center_col.col_name: cluster.center_point,
        }
        for cluster in clusters
    ]
    db_manager.store_in_table(Tables.cluster_attributes_table, attributes_row_dicts)

    # Store embeddings in embeddings table of DB
    embeddings_row_dicts = []
    for cluster in clusters:
        embeddings_row = [
            {Columns.cluster_id_col.col_name: cluster.cluster_id,
             Columns.embedding_col.col_name: embedding,
             Columns.face_id_col.col_name: face_id}
            for face_id, embedding in cluster.get_embeddings(with_embedding_ids=True)
        ]
        embeddings_row_dicts.extend(embeddings_row)
    db_manager.store_in_table(Tables.embeddings_table, embeddings_row_dicts)


def faces_to_embeddings(faces):
    # TODO: Finish implementing (what's missing?)
    for face in faces:
        yield Models.resnet(_to_tensor(face))


def user_choose_imgs(db_manager):
    # TODO: Refactor! (too many different tasks, function name non-descriptive)
    # TODO: Finish implementing (What's missing?)
    # TODO: Make user choose path
    # TODO: Disable dropping of existing tables
    # user_choose_path()
    path = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\Programming\BA\Logic\my_test\facenet_Test\group_imgs'
    db_manager.create_tables(create_local=True,
                             path_to_local_db=DBManager.get_db_path(path, local=True),
                             drop_existing_tables=True)
    faces_with_ids = extract_faces(path, db_manager)
    return faces_with_ids


def user_choose_path():
    path = input('Please enter a path with images of people you would like to add.\n')
    while not os.path.exists(path):
        log_error(f"Unable to find path '{path}'")
        print("\nPlease try again.")
        path = input('Please enter a path with images of people you would like to add.\n')
    return path  # IMG_PATH


def extract_faces(path, db_manager: DBManager):
    # TODO: Finish implementing (what's missing?)
    # TODO: Implement DB interactions
    # TODO: Generate Thumbnails differently? (E.g. via Image.thumbnail or sth. like that)
    # TODO: Store + update max_img_id and max_face_id somewhere rather than (always) get them via DB query?
    # TODO: Outsource db interactions to input-output logic?
    # TODO: Check if max_face_id maths checks out!

    path_to_local_db = DBManager.get_db_path(path, local=True)
    # Note: 'MAX' returns None / (None, ) as a default value
    max_img_id = db_manager.get_max_num(table=Tables.images_table, col=Columns.image_id_col, default=0,
                                        path_to_local_db=path_to_local_db)
    first_max_face_id = db_manager.get_max_num(table=Tables.embeddings_table, col=Columns.face_id_col, default=0)
    max_face_id = first_max_face_id

    faces = []
    img_loader = load_imgs_from_path(path, output_file_names=True, output_file_paths=True)
    for img_id, (img_path, img_name, img) in enumerate(img_loader, start=max_img_id+1):
        img_faces = cut_out_faces(Models.mtcnn, img)
        faces.extend(img_faces)
        last_modified = datetime.datetime.fromtimestamp(round(os.stat(img_path).st_mtime))
        img_row = {Columns.image_id_col.col_name: img_id,
                   Columns.file_name_col.col_name: img_name,
                   Columns.last_modified_col.col_name: last_modified}
        db_manager.store_in_table(Tables.images_table, [img_row], path_to_local_db)
        faces_rows = [{Columns.thumbnail_col.col_name: face,
                       Columns.image_id_col.col_name: img_id,
                       Columns.face_id_col.col_name: face_id}
                      for face_id, face in enumerate(img_faces, start=max_face_id+1)]
        max_face_id += len(img_faces)
        db_manager.store_in_table(Tables.faces_table, faces_rows, path_to_local_db)

    return enumerate(faces, start=first_max_face_id+1)


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


# def load_img_embeddings_from_dir(dir_path, output_file_name=False):
#     """
#     Yield all images in the given directory.
#     If img_img_extensions is empty, all files are assumed to be images. Otherwise, only files with extensions appearing
#     in the set will be returned.
#
#     :param output_file_name: Whether the tensor should be yielded together with the corresponding file name
#     :param dir_path: Directory containing images
#     :return: Yield(!) tuples of image_names and embeddings contained in this folder
#     """
#     # :param img_extensions: Set of lower-case file extensions considered images, e.g. {'jpg', 'png', 'gif'}. Empty = no
#     # filtering
#     # TODO: Needed?
#     # TODO: Finish implementing
#     if not output_file_name:
#         for img_name in get_img_names(dir_path):
#             with Image.open(dir_path + os.path.sep + img_name) as img:
#                 yield _to_tensor(img)
#     else:
#         for img_name in get_img_names(dir_path):
#             with Image.open(dir_path + os.path.sep + img_name) as img:
#                 yield img_name, _to_tensor(img)


def _to_tensor(img):
    return TO_TENSOR(img).unsqueeze(0)


def get_img_names(dir_path, img_extensions=None):
    """
    Yield all image file paths in dir_path.
    """
    # TODO: Finish implementing
    # TODO: Implement recursive option?
    obj_paths = map(lambda obj_name: os.path.join(dir_path, obj_name), os.listdir(dir_path))
    image_paths = filter(lambda obj_path: is_img(obj_path, img_extensions), obj_paths)
    return image_paths


def is_img(obj_path, img_extensions=None):
    """
    
    @param obj_path: Path to an object
    @param img_extensions: Iterable of extensions. Default: 'jpg', 'jpeg' and 'png'.
    @return: Whether obj_path ends with 
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


# --- i/o helpers ---

def _user_choose_cluster(clusters_path, return_names=True):
    clusters_names_and_paths = list(get_clusters_gen(clusters_path, return_names=True))
    clusters_names = get_every_nth_item(clusters_names_and_paths, n=0)
    prompt_cluster_choice(clusters_names)
    chosen_cluster_name = input()
    while chosen_cluster_name not in clusters_names:
        log_error(f'cluster "{chosen_cluster_name}" not found; Please try again.')
        prompt_cluster_choice(clusters_names)
        chosen_cluster_name = input()

    chosen_cluster_path = next(filter(lambda iterable: iterable[0] == chosen_cluster_name, clusters_names_and_paths))[1]
    if return_names:
        return chosen_cluster_name, chosen_cluster_path
    return chosen_cluster_path


def prompt_cluster_choice(clusters_names):
    # TODO: print limited number of clusters at a time (Enter=continue)
    temp_lim = 10
    clusters_names = clusters_names[:temp_lim]  # TODO: remove this line
    clusters_str = '\n'.join(map(lambda string: f'- {string}', clusters_names))
    wait_for_any_input('Which cluster would you like to view? (Press any key to continue.)')
    print(clusters_str)


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
    'add': handler_add_new_embeddings,
    'showcluster': handler_show_cluster,
}

handlers_params = {
    'showcluster': [],
}
